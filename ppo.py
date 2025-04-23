import wandb
import gymnasium as gym
import imageio
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim

from dataclasses import dataclass
from gymnasium.wrappers import TransformObservation
from gymnasium.spaces import Box

ACTION_SCALE = 1.0

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
    def forward(self, x):
        return x + self.block(x)

class CNN(nn.Module):
  def __init__(self, obs_shape, hidden):
    super().__init__()
    C, H, W = obs_shape
    assert H == 96 and W == 96, "wrong input size"
    self.conv = nn.Sequential(
      nn.Conv2d(C, 32, 8, 4), nn.ReLU(), ResidualBlock(32),
      nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(), ResidualBlock(64),
      nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), ResidualBlock(64),
    )
    conv_out = self._get_conv_out(obs_shape)
    self.fc = nn.Linear(conv_out, hidden)
  def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv(o)
        return int(torch.prod(torch.tensor(o.shape[1:])))
  def forward(self, x):
    x = x.float() / 255.0
    x = self.conv(x)
    x = x.view(x.size(0), -1)
    return self.fc(x)


LOG_STD_MIN, LOG_STD_MAX = -7, 2

def atanh(x):
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim, hidden=256):
        super().__init__()
        self.enc = CNN(obs_shape, hidden)
        self.mu = nn.Linear(hidden, action_dim)
        self.action_logstd = nn.Parameter(torch.ones(action_dim) * 0.0)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs):
        h = self.enc(obs)
        mu = self.mu(h) * ACTION_SCALE

        log_std = self.action_logstd.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        std = std.expand_as(mu)

        dist = torch.distributions.Normal(mu, std)
        pre_tanh = dist.rsample()
        action = torch.tanh(pre_tanh)

        logp = dist.log_prob(pre_tanh) - torch.log(1 - action.pow(2) + 1e-6)
        logp = logp.sum(-1, keepdim=True)

        value = self.v(h).squeeze(-1)
        return action, logp, value, mu, log_std

    def log_prob(self, obs, action):
        h = self.enc(obs)
        mu = self.mu(h) * ACTION_SCALE

        log_std = self.action_logstd.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        std = std.expand_as(mu)

        dist = torch.distributions.Normal(mu, std)
        pre_tanh = atanh(action.clamp(-0.999, 0.999))
        logp = dist.log_prob(pre_tanh) - torch.log(1 - action.pow(2) + 1e-6)
        return logp.sum(-1, keepdim=True), self.v(h).squeeze(-1)

class Rollout:
    def __init__(self, hp, obs_shape, action_dim):
        S, N = hp.rollout_len, hp.num_envs
        self.obs = torch.zeros(S, N, *obs_shape, dtype=torch.uint8)
        self.actions = torch.zeros(S, N, action_dim)
        self.logp = torch.zeros(S, N, 1)
        self.rewards = torch.zeros(S, N)
        self.values = torch.zeros(S, N)
        self.dones = torch.zeros(S, N, dtype=torch.bool)
        self.env_acts = torch.zeros(S, N, action_dim)
        self.mus = torch.zeros(S, N, action_dim)
        self.log_stds = torch.zeros(S, N, action_dim)
    def add(self, t, obs, action, logp, reward, value, done, env_act, mu, std):
        self.obs[t].copy_(obs.cpu())
        self.actions[t] = action.cpu()
        self.logp[t] = logp.cpu()
        self.rewards[t] = reward
        self.values[t] = value.cpu()
        self.dones[t] = done
        self.env_acts[t] = env_act
        self.mus[t] = mu
        self.log_stds[t] = std
    def compute_adv_returns(self, last_val, hp):
        adv = torch.zeros_like(self.rewards)
        last_gae = 0
        for t in reversed(range(hp.rollout_len)):
            mask = 1.0 - self.dones[t].float()
            next_val = self.values[t+1] if t < hp.rollout_len-1 else last_val
            delta = self.rewards[t] + hp.gamma * next_val * mask - self.values[t]
            last_gae = delta + hp.gamma * hp.gae_lambda * mask * last_gae
            adv[t] = last_gae
        ret = adv + self.values
        return adv.flatten().to(hp.device), ret.flatten().to(hp.device)
    def get_action_means(self):
      mask = (~self.dones).float()
      mask = mask.unsqueeze(-1)
      masked_actions = self.env_acts * mask

      total_valid = mask.sum(dim=(0, 1))
      return masked_actions.sum(dim=(0, 1)) / (total_valid + 1e-8)
    def get_reward_means(self):
      mask = (~self.dones).float()
      mask = mask.unsqueeze(-1)
      masked_rewards = self.rewards.unsqueeze(-1) * mask
      return masked_rewards.sum()
    def get_policy_stats(self):
      mask = (~self.dones).float()
      mask = mask.unsqueeze(-1)
      masked_mus = self.mus * mask
      masked_log_stds = self.log_stds * mask

      total_valid = mask.sum(dim=(0,1))
      return masked_mus.sum(dim=(0,1)) / (total_valid + 1e-8), masked_log_stds.sum(dim=(0,1)) / (total_valid + 1e-8)

@dataclass
class HP:
  env_name: str = 'CarRacing-v3'
  num_envs: int = 256
  rollout_len: int = 64
  epochs: int = 4
  minibatches: int = 128
  gamma: float = 0.99
  gae_lambda: float = 0.85
  clip_eps: float = 0.2
  ppo_clip_eps: float = 0.2
  vf_coef: float = 0.1
  ent_coef: float = 0.001
  lr: float = 5e-5
  total_frames: int = 100_000_000
  device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_env():
  env = gym.make(HP.env_name, continuous=True, domain_randomize=False)
  orig_space = env.observation_space
  low = orig_space.low.transpose(2,0,1)
  high = orig_space.high.transpose(2,0,1)
  new_space = Box(low=low, high=high, dtype=orig_space.dtype)
  return TransformObservation(env, lambda o: o.transpose(2,0,1), observation_space=new_space)

def transform_action(a, brake_threshold=0.05):
    a = a.clone()
    # a[:, 1] = (a[:, 1] + 1) / 2
    # a[:, 2] = (a[:, 2] + 1) / 2
    a[:, 2][a[:, 2] <= brake_threshold] = 0.0
    return a

def evaluate_greedy(net, episodes=1, max_steps=500, record_path=None):

    render_mode = 'rgb_array' if record_path else None
    env = gym.make(HP.env_name, continuous=True, domain_randomize=False, render_mode=render_mode)
    env = TransformObservation(env, lambda o: o.transpose(2, 0, 1), observation_space=env.observation_space)

    scores = []
    frames = [] if record_path else None
    actions = []

    for ep in range(episodes):
        obs, _ = env.reset()
        ep_ret, done = 0.0, False
        steps = 0
        while not done and steps < max_steps:
            obs_t = torch.tensor(obs, dtype=torch.uint8).unsqueeze(0).to(HP.device)
            with torch.no_grad():
                _, _, _, mu, _ = net(obs_t)
                act = torch.tanh(mu)


            action = transform_action(act).squeeze(0).cpu().numpy().astype(np.float32)
            # action = torch.tensor([[1.0, 0.3, 0.2]])
            # action = np.array([1.0, 0.3, 0.2], dtype=np.float32)
            # action = transform_action(action).squeeze(0).cpu().numpy().astype(np.float32)
            actions.append(action)
            obs, r, done, _, _ = env.step(action)
            ep_ret += r
            steps += 1

            if record_path and ep == 0:
                frame = env.render()
                frames.append(frame)

        scores.append(ep_ret)

    env.close()

    if record_path and frames:
        imageio.mimwrite(record_path, frames, fps=30)

    avg_steer, avg_gas, avg_brake = np.mean(actions, axis=0).tolist()
    return float(np.mean(scores)), avg_steer, avg_gas, avg_brake


def main():
    wandb.init(project="ppo-carracing", config={
      "env_name": HP.env_name,
      "num_envs": HP.num_envs,
      "rollout_len": HP.rollout_len,
      "epochs": HP.epochs,
      "gamma": HP.gamma,
      "gae_lambda": HP.gae_lambda,
      "clip_eps": HP.clip_eps,
      "ppo_clip_eps": HP.ppo_clip_eps,
      "vf_coef": HP.vf_coef,
      "ent_coef": HP.ent_coef,
      "lr": HP.lr,
      "total_frames": HP.total_frames,
    })
    hp = HP()
    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(hp.num_envs)])
    obs_shape = envs.single_observation_space.shape
    action_dim = int(np.prod(envs.single_action_space.shape))

    net = ActorCritic(obs_shape, action_dim).to(hp.device)
    optimizer = optim.Adam(net.parameters(), lr = hp.lr, eps=1e-5)

    global_frames = 0
    eval_interval = 500_000
    next_eval = eval_interval

    obs, _ = envs.reset()

    while global_frames < hp.total_frames:
      policy_loss_mem = []
      value_loss_mem = []
      entropy_mem = []
      total_loss_mem = []

      if global_frames >= next_eval:
        print('start evaluating')
        avg_ret, avg_steer, avg_gas, avg_brake = evaluate_greedy(net, 1, 2000, record_path=f"last/{next_eval}.mp4")
        wandb.log({"eval/avg_return": avg_ret}, step=global_frames)
        wandb.log({"eval/avg_steer": avg_steer}, step=global_frames)
        wandb.log({"eval/avg_gas": avg_gas}, step=global_frames)
        wandb.log({"eval/avg_brake": avg_brake}, step=global_frames)

        print(f"[eval] frames {global_frames/1e6:.2f}M  avg_return {avg_ret:.1f}")
        next_eval += eval_interval

      roll = Rollout(hp, obs_shape, action_dim)

      for t in range(hp.rollout_len):
        obs_t = torch.tensor(obs, dtype=torch.uint8, device=hp.device)
        with torch.no_grad():
          act, logp, val, mu, log_std = net(obs_t)
          env_act = transform_action(act).cpu().numpy()
          # if global_frames < 5000_000:
          #   env_act[:, 2] = 0.0
            # print('warmup roll out', env_act)
          next_obs, reward, done, _, _ = envs.step(env_act)
          roll.add(t, obs_t.cpu(), act.cpu(), logp.cpu(), torch.tensor(reward), val.cpu(), torch.tensor(done), torch.tensor(env_act), mu.cpu(), log_std.cpu())
          obs = next_obs
          global_frames += hp.num_envs
      with torch.no_grad():
        last_obs = torch.tensor(obs, dtype = torch.uint8, device=hp.device)
        _, _, last_val, _, _ = net(last_obs)
        last_val = last_val.cpu()
      
      adv, ret = roll.compute_adv_returns(last_val, hp)
      b_obs = roll.obs.view(-1, *obs_shape).float().to(hp.device)
      b_actions = roll.actions.view(-1, action_dim).to(hp.device)
      b_logp_old = roll.logp.view(-1, 1).to(hp.device)
      b_adv = (adv - adv.mean()) / (adv.std() + 1e-8)
      b_ret = ret      

      inds = np.arange(len(b_adv))
      for _ in range(hp.epochs):
        np.random.shuffle(inds)
        mb_size = len(b_adv) // hp.minibatches
        for start in range(0, len(b_adv), mb_size):
          mb_inds = inds[start:start+mb_size]
          obs_mb = b_obs[mb_inds]
          act_mb = b_actions[mb_inds]
          logp_old_mb = b_logp_old[mb_inds]
          adv_mb = b_adv[mb_inds]
          ret_mb = b_ret[mb_inds]

          logp_new, value = net.log_prob(obs_mb, act_mb)
          ratio = (logp_new - logp_old_mb).exp()
          surr1 = ratio * adv_mb
          surr2 = torch.clamp(ratio, 1-hp.ppo_clip_eps, 1+hp.ppo_clip_eps) * adv_mb
          policy_loss = -torch.min(surr1, surr2).mean()

          value_pred_clipped = roll.values.view(-1)[mb_inds].to(hp.device) + (value - roll.values.view(-1)[mb_inds].to(hp.device)).clamp(-hp.clip_eps, hp.clip_eps)
          value_losses = (value - ret_mb).pow(2)
          value_losses_clipped = (value_pred_clipped - ret_mb).pow(2)
          value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

          entropy = (-logp_new).mean()
          entropy_per_dim = entropy.item() / action_dim
          loss = policy_loss + hp.vf_coef*value_loss - hp.ent_coef*entropy
          policy_loss_mem.append(policy_loss)
          value_loss_mem.append(value_loss)
          entropy_mem.append(entropy_per_dim)
          total_loss_mem.append(loss)


          # optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5); optimizer.step()
          optimizer.zero_grad(); loss.backward(); optimizer.step()
      
      roll_act_0, roll_act_1, roll_act_2 = roll.get_action_means().cpu().tolist()
      roll_reward = roll.get_reward_means().cpu().tolist()
      roll_mus, roll_stds = roll.get_policy_stats()
      roll_mu, _, _ = roll_mus.cpu().tolist()
      roll_std, _, _ = roll_stds.cpu().tolist()
      # print(roll_act_0)
      entropy_mem = np.array(entropy_mem)
      policy_loss_mean = torch.stack(policy_loss_mem).mean().item()
      value_loss_mean = torch.stack(value_loss_mem).mean().item()
      total_loss_mean = torch.stack(total_loss_mem).mean().item()


      wandb.log({
        "global_frames": global_frames,
        "policy_loss": policy_loss_mean,
        "value_loss": value_loss_mean,
        "entropy": entropy_mem.mean().item(),
        "loss": total_loss_mean,
        "avg_advantage": adv.mean().item(),
        "frames": global_frames,
        "act_0": roll_act_0,
        "act_1": roll_act_1,
        "act_2": roll_act_2,
        "mu": roll_mu,
        "log_std": roll_std
      }, step=global_frames)

      if global_frames % 100_000 == 0:
        avg_adv = adv.mean().item()
        print(
            f"Frames {global_frames/1e6:.2f}M  "
            f"avg_adv {avg_adv:.3f}  "
            f"policy_loss {policy_loss.item():.3f}  "
            f"value_loss {value_loss.item():.3f}  "
            f"entropy {entropy_per_dim:.3f():.3f} "
          )


if __name__ == '__main__':
  main()