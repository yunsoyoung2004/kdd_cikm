#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ============================================================
# robust_multi_env_recurrent_ppo.py
# ------------------------------------------------------------
# 목표
# - CartPole / Acrobot / MountainCar 에서 안정적으로 학습되는 recurrent PPO
# - sparse reward 환경 대응
# - mild / medium / hard POMDP curriculum 지원
# - notebook / terminal 모두 안전
#
# 주요 개선
# - parse_known_args() : notebook 안전
# - recurrent PPO (LSTM)
# - orthogonal init + LayerNorm
# - observation normalization
# - PPO value clipping / KL early stopping
# - lr annealing / entropy annealing
# - env-specific shaping reward (train only)
# - raw reward / train reward 분리
# - scheduled POMDP severity
# - checkpoint / resume / graceful stop
# - optional notebook kernel safe shutdown
#
# 권장 실행
# python robust_multi_env_recurrent_ppo.py --device cuda:0 --resume
# ============================================================

import os
import gc
import csv
import json
import math
import time
import signal
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ============================================================
# Global
# ============================================================

USE_AMP_DEFAULT = True
STOP_REQUESTED = False


def _signal_handler(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"\n[signal] received signal={signum}. stopping at next safe point...")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ============================================================
# Utils
# ============================================================

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def cleanup_torch(device: torch.device):
    try:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception as e:
        print("[warn] cleanup_torch failed:", repr(e))


def unwrap_model(model: nn.Module) -> nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def model_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return unwrap_model(model).state_dict()


def load_model_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor], strict: bool = True):
    unwrap_model(model).load_state_dict(state_dict, strict=strict)


def maybe_compile(model: nn.Module, enabled: bool) -> nn.Module:
    if not enabled:
        return model
    try:
        return torch.compile(model)
    except Exception as e:
        print("[warn] torch.compile failed, fallback:", repr(e))
        return model


def linear_interp(a: float, b: float, t01: float) -> float:
    t01 = max(0.0, min(1.0, float(t01)))
    return a + (b - a) * t01


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def maybe_shutdown_kernel(enabled: bool):
    if not enabled:
        return
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None and hasattr(ip, "kernel"):
            print("[shutdown] notebook kernel will be shut down safely.")
            ip.kernel.do_shutdown(restart=False)
    except Exception as e:
        print("[warn] kernel shutdown skipped:", repr(e))


# ============================================================
# Running mean/std
# ============================================================

class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-12)
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        z = (x - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(z, -10.0, 10.0).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "count": float(self.count),
        }

    def load_state_dict(self, sd: Dict[str, Any]):
        self.mean = np.asarray(sd["mean"], dtype=np.float64)
        self.var = np.asarray(sd["var"], dtype=np.float64)
        self.count = float(sd["count"])


# ============================================================
# POMDP wrapper with curriculum
# ============================================================

class ScheduledPartialObsWrapper(gym.ObservationWrapper):
    """
    partial_indices: 일부 차원만 관측
    flicker/noise는 progress에 따라 start -> end
    """
    def __init__(
        self,
        env: gym.Env,
        partial_indices: Optional[List[int]],
        flicker_prob_start: float,
        flicker_prob_end: float,
        noise_std_start: float,
        noise_std_end: float,
    ):
        super().__init__(env)
        self.partial_indices = partial_indices
        self.flicker_prob_start = float(flicker_prob_start)
        self.flicker_prob_end = float(flicker_prob_end)
        self.noise_std_start = float(noise_std_start)
        self.noise_std_end = float(noise_std_end)
        self.progress = 0.0

        assert isinstance(env.observation_space, gym.spaces.Box), "Only Box observation supported"
        low = env.observation_space.low
        high = env.observation_space.high

        if self.partial_indices is None:
            self.partial_indices = list(range(int(np.prod(low.shape))))

        low2 = low[self.partial_indices].astype(np.float32)
        high2 = high[self.partial_indices].astype(np.float32)

        self.observation_space = gym.spaces.Box(
            low=low2,
            high=high2,
            shape=(len(self.partial_indices),),
            dtype=np.float32,
        )

    def set_progress(self, progress01: float):
        self.progress = max(0.0, min(1.0, float(progress01)))

    def get_current_params(self):
        flicker_prob = linear_interp(self.flicker_prob_start, self.flicker_prob_end, self.progress)
        noise_std = linear_interp(self.noise_std_start, self.noise_std_end, self.progress)
        return flicker_prob, noise_std

    def observation(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        obs = obs[self.partial_indices].astype(np.float32)

        flicker_prob, noise_std = self.get_current_params()

        if flicker_prob > 0.0 and np.random.rand() < flicker_prob:
            obs = np.zeros_like(obs, dtype=np.float32)

        if noise_std > 0.0:
            obs = obs + np.random.normal(0.0, noise_std, size=obs.shape).astype(np.float32)

        return obs.astype(np.float32)


# ============================================================
# Reward shaping
# ============================================================

class RewardShaper:
    def reset(self, obs: np.ndarray):
        pass

    def train_reward(self, obs, action, next_obs, raw_reward, terminated, truncated):
        return float(raw_reward)


class IdentityShaper(RewardShaper):
    pass


class CartPoleShaper(RewardShaper):
    """
    CartPole은 원래 dense reward라 shaping 거의 안 함.
    너무 과한 shaping은 오히려 망가질 수 있으니 작은 안정화만.
    """
    def train_reward(self, obs, action, next_obs, raw_reward, terminated, truncated):
        # obs: [x, x_dot, theta, theta_dot]
        theta = abs(float(next_obs[2])) if len(next_obs) >= 3 else 0.0
        x = abs(float(next_obs[0])) if len(next_obs) >= 1 else 0.0

        # 작은 penalty만 추가
        bonus = 0.05 * max(0.0, 0.2 - theta) - 0.01 * x
        return float(raw_reward + bonus)


class MountainCarShaper(RewardShaper):
    """
    potential-based shaping:
    높이 + 진행 방향 + 정상 종료 bonus
    """
    def potential(self, obs):
        pos = float(obs[0])
        vel = float(obs[1])
        # position, forward motion
        return 5.0 * pos + 1.0 * vel

    def train_reward(self, obs, action, next_obs, raw_reward, terminated, truncated):
        gamma = 0.99
        shaped = float(raw_reward) + gamma * self.potential(next_obs) - self.potential(obs)

        # goal reached bonus
        if terminated and float(next_obs[0]) >= 0.5:
            shaped += 10.0
        return shaped


class AcrobotShaper(RewardShaper):
    """
    potential-based shaping:
    end-effector 높이 기반
    """
    def tip_height(self, obs):
        c1, s1, c2, s2 = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])
        theta1 = math.atan2(s1, c1)
        theta2 = math.atan2(s2, c2)
        # link length = 1,1
        y = -math.cos(theta1) - math.cos(theta1 + theta2)
        return y

    def potential(self, obs):
        # tip 높이가 위로 갈수록 좋음
        return 4.0 * self.tip_height(obs)

    def train_reward(self, obs, action, next_obs, raw_reward, terminated, truncated):
        gamma = 0.99
        shaped = float(raw_reward) + gamma * self.potential(next_obs) - self.potential(obs)
        if terminated:
            shaped += 10.0
        return shaped


# ============================================================
# Presets
# ============================================================

@dataclass
class EnvPreset:
    name: str
    env_id: str
    partial_indices: Optional[List[int]]

    flicker_prob_start: float
    flicker_prob_end: float
    noise_std_start: float
    noise_std_end: float

    max_steps: int
    solve_score: Optional[float]
    recommended_episodes: int

    lr: float
    lr_end_scale: float
    gamma: float
    gae_lambda: float
    ppo_clip: float
    value_clip: float
    ppo_epochs: int
    minibatch_tokens: int
    steps_per_update: int
    value_coef: float
    max_grad_norm: float
    target_kl: float

    ent_start: float
    ent_end: float

    mlp_hidden: int
    lstm_hidden: int

    shaper: str


def get_presets() -> Dict[str, EnvPreset]:
    presets = {
        # ----------------------------------------------------
        # RELIABLE
        # ----------------------------------------------------
        "cartpole_reliable": EnvPreset(
            name="cartpole_reliable",
            env_id="CartPole-v1",
            partial_indices=[0, 1, 2, 3],
            flicker_prob_start=0.00,
            flicker_prob_end=0.03,
            noise_std_start=0.0,
            noise_std_end=0.003,
            max_steps=500,
            solve_score=475.0,
            recommended_episodes=4000,
            lr=3e-4,
            lr_end_scale=0.20,
            gamma=0.99,
            gae_lambda=0.95,
            ppo_clip=0.2,
            value_clip=0.2,
            ppo_epochs=4,
            minibatch_tokens=1024,
            steps_per_update=1024,
            value_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.03,
            ent_start=0.02,
            ent_end=0.001,
            mlp_hidden=128,
            lstm_hidden=128,
            shaper="cartpole",
        ),

        "acrobot_reliable": EnvPreset(
            name="acrobot_reliable",
            env_id="Acrobot-v1",
            partial_indices=[0, 1, 2, 3, 4, 5],
            flicker_prob_start=0.00,
            flicker_prob_end=0.03,
            noise_std_start=0.0,
            noise_std_end=0.001,
            max_steps=500,
            solve_score=-100.0,
            recommended_episodes=16000,
            lr=2.5e-4,
            lr_end_scale=0.15,
            gamma=0.99,
            gae_lambda=0.97,
            ppo_clip=0.2,
            value_clip=0.2,
            ppo_epochs=8,
            minibatch_tokens=2048,
            steps_per_update=4096,
            value_coef=0.6,
            max_grad_norm=0.7,
            target_kl=0.025,
            ent_start=0.02,
            ent_end=0.002,
            mlp_hidden=256,
            lstm_hidden=128,
            shaper="acrobot",
        ),

        "mountaincar_reliable": EnvPreset(
            name="mountaincar_reliable",
            env_id="MountainCar-v0",
            partial_indices=[0, 1],
            flicker_prob_start=0.00,
            flicker_prob_end=0.02,
            noise_std_start=0.0,
            noise_std_end=0.0005,
            max_steps=200,
            solve_score=-110.0,
            recommended_episodes=18000,
            lr=2e-4,
            lr_end_scale=0.10,
            gamma=0.99,
            gae_lambda=0.97,
            ppo_clip=0.2,
            value_clip=0.2,
            ppo_epochs=10,
            minibatch_tokens=2048,
            steps_per_update=4096,
            value_coef=0.7,
            max_grad_norm=0.7,
            target_kl=0.02,
            ent_start=0.03,
            ent_end=0.003,
            mlp_hidden=256,
            lstm_hidden=128,
            shaper="mountaincar",
        ),

        # ----------------------------------------------------
        # MEDIUM POMDP
        # ----------------------------------------------------
        "cartpole_medium": EnvPreset(
            name="cartpole_medium",
            env_id="CartPole-v1",
            partial_indices=[0, 2, 3],
            flicker_prob_start=0.01,
            flicker_prob_end=0.06,
            noise_std_start=0.001,
            noise_std_end=0.01,
            max_steps=500,
            solve_score=475.0,
            recommended_episodes=6000,
            lr=3e-4,
            lr_end_scale=0.20,
            gamma=0.99,
            gae_lambda=0.95,
            ppo_clip=0.2,
            value_clip=0.2,
            ppo_epochs=5,
            minibatch_tokens=1024,
            steps_per_update=1024,
            value_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.03,
            ent_start=0.02,
            ent_end=0.001,
            mlp_hidden=128,
            lstm_hidden=128,
            shaper="cartpole",
        ),

        "acrobot_medium": EnvPreset(
            name="acrobot_medium",
            env_id="Acrobot-v1",
            partial_indices=[0, 1, 2, 3, 5],
            flicker_prob_start=0.01,
            flicker_prob_end=0.05,
            noise_std_start=0.0,
            noise_std_end=0.003,
            max_steps=500,
            solve_score=-100.0,
            recommended_episodes=22000,
            lr=2.5e-4,
            lr_end_scale=0.15,
            gamma=0.99,
            gae_lambda=0.97,
            ppo_clip=0.2,
            value_clip=0.2,
            ppo_epochs=8,
            minibatch_tokens=2048,
            steps_per_update=4096,
            value_coef=0.6,
            max_grad_norm=0.7,
            target_kl=0.025,
            ent_start=0.02,
            ent_end=0.002,
            mlp_hidden=256,
            lstm_hidden=128,
            shaper="acrobot",
        ),

        "mountaincar_medium": EnvPreset(
            name="mountaincar_medium",
            env_id="MountainCar-v0",
            partial_indices=[0, 1],
            flicker_prob_start=0.01,
            flicker_prob_end=0.05,
            noise_std_start=0.0002,
            noise_std_end=0.003,
            max_steps=200,
            solve_score=-110.0,
            recommended_episodes=24000,
            lr=2e-4,
            lr_end_scale=0.10,
            gamma=0.99,
            gae_lambda=0.97,
            ppo_clip=0.2,
            value_clip=0.2,
            ppo_epochs=10,
            minibatch_tokens=2048,
            steps_per_update=4096,
            value_coef=0.7,
            max_grad_norm=0.7,
            target_kl=0.02,
            ent_start=0.03,
            ent_end=0.003,
            mlp_hidden=256,
            lstm_hidden=128,
            shaper="mountaincar",
        ),

        # ----------------------------------------------------
        # HARD
        # ----------------------------------------------------
        "cartpole_partial_hard": EnvPreset(
            name="cartpole_partial_hard",
            env_id="CartPole-v1",
            partial_indices=[0, 2],
            flicker_prob_start=0.02,
            flicker_prob_end=0.10,
            noise_std_start=0.001,
            noise_std_end=0.02,
            max_steps=500,
            solve_score=475.0,
            recommended_episodes=9000,
            lr=3e-4,
            lr_end_scale=0.20,
            gamma=0.99,
            gae_lambda=0.95,
            ppo_clip=0.2,
            value_clip=0.2,
            ppo_epochs=6,
            minibatch_tokens=1024,
            steps_per_update=1024,
            value_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.025,
            ent_start=0.02,
            ent_end=0.001,
            mlp_hidden=128,
            lstm_hidden=128,
            shaper="cartpole",
        ),
    }
    return presets


def get_shaper(name: str) -> RewardShaper:
    name = str(name).lower()
    if name == "cartpole":
        return CartPoleShaper()
    if name == "acrobot":
        return AcrobotShaper()
    if name == "mountaincar":
        return MountainCarShaper()
    return IdentityShaper()


def build_env(preset: EnvPreset, seed: Optional[int] = None):
    env = gym.make(preset.env_id)
    env = ScheduledPartialObsWrapper(
        env=env,
        partial_indices=preset.partial_indices,
        flicker_prob_start=preset.flicker_prob_start,
        flicker_prob_end=preset.flicker_prob_end,
        noise_std_start=preset.noise_std_start,
        noise_std_end=preset.noise_std_end,
    )
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


# ============================================================
# Model
# ============================================================

class RecurrentActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, mlp_hidden: int, lstm_hidden: int):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.mlp_hidden = mlp_hidden
        self.lstm_hidden = lstm_hidden

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.Tanh(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.Tanh(),
        )

        self.lstm = nn.LSTM(
            input_size=mlp_hidden,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.actor = nn.Linear(lstm_hidden, action_dim)
        self.critic = nn.Linear(lstm_hidden, 1)

        self.apply(self._init_weights)

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            nn.init.zeros_(m.bias)

    def initial_state(self, batch_size: int, device: torch.device):
        h = torch.zeros((1, batch_size, self.lstm_hidden), device=device)
        c = torch.zeros((1, batch_size, self.lstm_hidden), device=device)
        return (h, c)

    def forward(self, obs_seq: torch.Tensor, hidden=None):
        # obs_seq: (B,T,D)
        x = self.encoder(obs_seq)
        x, hidden = self.lstm(x, hidden)
        logits = self.actor(x)
        values = self.critic(x)
        return logits, values, hidden

    @torch.no_grad()
    def act(self, obs_t: torch.Tensor, hidden):
        logits, value, hidden = self.forward(obs_t, hidden)
        logits = logits[:, -1, :]
        value = value[:, -1, :]
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.item()),
            float(entropy.item()),
            hidden,
        )


# ============================================================
# Rollout buffer
# ============================================================

class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.train_rewards = []
        self.raw_rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.episode_ids = []
        self.bootstrap_values = []

    def clear(self):
        self.__init__()

    def add(self, obs, action, train_reward, raw_reward, done, log_prob, value, episode_id):
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.actions.append(int(action))
        self.train_rewards.append(float(train_reward))
        self.raw_rewards.append(float(raw_reward))
        self.dones.append(bool(done))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.episode_ids.append(int(episode_id))

    def __len__(self):
        return len(self.obs)


# ============================================================
# Metrics / GAE / segments
# ============================================================

def compute_train_metrics(raw_rewards: List[float], solve_score: Optional[float]):
    r = np.asarray(raw_rewards, dtype=np.float32)
    if len(r) == 0:
        return {
            "auc_mean": 0.0,
            "last100_mean": 0.0,
            "last100_std": 0.0,
            "best_reward": 0.0,
            "first_solve_ep": -1,
        }

    tail = r[-100:] if len(r) >= 100 else r
    last100_mean = float(tail.mean())
    last100_std = float(tail.std(ddof=0))
    best_reward = float(r.max())
    auc_mean = float(r.mean())

    first_solve_ep = -1
    if solve_score is not None and len(r) >= 100:
        win = np.convolve(r, np.ones(100, dtype=np.float32) / 100.0, mode="valid")
        idx = np.where(win >= solve_score)[0]
        if len(idx) > 0:
            first_solve_ep = int(idx[0] + 100)

    return {
        "auc_mean": auc_mean,
        "last100_mean": last100_mean,
        "last100_std": last100_std,
        "best_reward": best_reward,
        "first_solve_ep": int(first_solve_ep),
    }


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: float,
    gamma: float,
    gae_lambda: float,
):
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        if t == n - 1:
            next_nonterminal = 1.0 - float(dones[t])
            next_values = next_value
        else:
            next_nonterminal = 1.0 - float(dones[t])
            next_values = values[t + 1]

        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def split_into_segments_preview(buffer: RolloutBuffer):
    if len(buffer) == 0:
        return []

    segments = []
    start = 0
    n = len(buffer)

    for i in range(n - 1):
        if buffer.dones[i] or buffer.episode_ids[i + 1] != buffer.episode_ids[i]:
            segments.append(slice(start, i + 1))
            start = i + 1

    if start < n:
        segments.append(slice(start, n))

    return segments


def split_into_segments(buffer: RolloutBuffer):
    previews = split_into_segments_preview(buffer)
    out = []
    for idx, sl in enumerate(previews):
        out.append((sl, buffer.bootstrap_values[idx]))
    return out


def pad_sequences(arrays: List[np.ndarray], pad_value: float = 0.0):
    max_len = max(a.shape[0] for a in arrays)

    padded = []
    masks = []
    for a in arrays:
        T = a.shape[0]
        pad_shape = (max_len - T,) + a.shape[1:]
        if len(pad_shape) == 1:
            pad_arr = np.full((max_len - T,), pad_value, dtype=a.dtype)
        else:
            pad_arr = np.full(pad_shape, pad_value, dtype=a.dtype)

        padded_arr = np.concatenate([a, pad_arr], axis=0)
        mask = np.concatenate(
            [np.ones(T, dtype=np.float32), np.zeros(max_len - T, dtype=np.float32)],
            axis=0,
        )
        padded.append(padded_arr)
        masks.append(mask)

    return np.stack(padded), np.stack(masks)


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_policy(
    model: nn.Module,
    preset: EnvPreset,
    device: torch.device,
    obs_rms: RunningMeanStd,
    n_episodes: int = 10,
    greedy: bool = True,
):
    env = build_env(preset, seed=12345)
    env.set_progress(1.0)  # hardest scheduled test
    scores = []

    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=12345 + ep)
            hidden = unwrap_model(model).initial_state(batch_size=1, device=device)
            done = False
            ep_reward = 0.0
            steps = 0

            while not done and steps < preset.max_steps:
                obs_n = obs_rms.normalize(obs)
                obs_t = torch.as_tensor(obs_n, dtype=torch.float32, device=device).view(1, 1, -1)
                logits, _, hidden = model(obs_t, hidden)
                logits = logits[:, -1, :]
                dist = Categorical(logits=logits)
                action = torch.argmax(dist.probs, dim=-1) if greedy else dist.sample()

                obs, reward, terminated, truncated, _ = env.step(int(action.item()))
                done = terminated or truncated
                ep_reward += float(reward)
                steps += 1

            scores.append(ep_reward)
    finally:
        env.close()

    arr = np.asarray(scores, dtype=np.float32)
    return {
        "eval_mean": float(arr.mean()) if len(arr) else 0.0,
        "eval_std": float(arr.std(ddof=0)) if len(arr) else 0.0,
        "eval_min": float(arr.min()) if len(arr) else 0.0,
        "eval_max": float(arr.max()) if len(arr) else 0.0,
    }


# ============================================================
# PPO update
# ============================================================

def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    buffer: RolloutBuffer,
    preset: EnvPreset,
    device: torch.device,
    use_amp: bool,
    progress01: float,
):
    segments = split_into_segments(buffer)
    if not segments:
        return {
            "loss_total": 0.0,
            "loss_policy": 0.0,
            "loss_value": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "early_stop_kl": 0,
        }

    obs_list = []
    act_list = []
    old_logp_list = []
    adv_list = []
    ret_list = []
    old_value_list = []

    for sl, bootstrap_value in segments:
        rewards = np.asarray(buffer.train_rewards[sl], dtype=np.float32)
        values = np.asarray(buffer.values[sl], dtype=np.float32)
        dones = np.asarray(buffer.dones[sl], dtype=np.float32)

        adv, ret = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            next_value=float(bootstrap_value),
            gamma=preset.gamma,
            gae_lambda=preset.gae_lambda,
        )

        obs_arr = np.asarray(buffer.obs[sl], dtype=np.float32)
        act_arr = np.asarray(buffer.actions[sl], dtype=np.int64)
        logp_arr = np.asarray(buffer.log_probs[sl], dtype=np.float32)
        val_arr = np.asarray(buffer.values[sl], dtype=np.float32)

        obs_list.append(obs_arr)
        act_list.append(act_arr)
        old_logp_list.append(logp_arr)
        adv_list.append(adv.astype(np.float32))
        ret_list.append(ret.astype(np.float32))
        old_value_list.append(val_arr.astype(np.float32))

    # advantage normalization
    all_adv = np.concatenate(adv_list, axis=0)
    adv_mean = all_adv.mean()
    adv_std = all_adv.std() + 1e-8
    adv_list = [((a - adv_mean) / adv_std).astype(np.float32) for a in adv_list]

    obs_pad, mask = pad_sequences(obs_list, pad_value=0.0)
    act_pad, _ = pad_sequences([a[:, None] for a in act_list], pad_value=0)
    old_logp_pad, _ = pad_sequences([a[:, None] for a in old_logp_list], pad_value=0.0)
    adv_pad, _ = pad_sequences([a[:, None] for a in adv_list], pad_value=0.0)
    ret_pad, _ = pad_sequences([a[:, None] for a in ret_list], pad_value=0.0)
    old_val_pad, _ = pad_sequences([a[:, None] for a in old_value_list], pad_value=0.0)

    obs_t = torch.as_tensor(obs_pad, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(act_pad.squeeze(-1), dtype=torch.long, device=device)
    old_logp_t = torch.as_tensor(old_logp_pad.squeeze(-1), dtype=torch.float32, device=device)
    adv_t = torch.as_tensor(adv_pad.squeeze(-1), dtype=torch.float32, device=device)
    ret_t = torch.as_tensor(ret_pad.squeeze(-1), dtype=torch.float32, device=device)
    old_val_t = torch.as_tensor(old_val_pad.squeeze(-1), dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(mask, dtype=torch.float32, device=device)

    n_seq = obs_t.shape[0]
    idxs = np.arange(n_seq)

    ent_coef = linear_interp(preset.ent_start, preset.ent_end, progress01)

    total_loss_v = []
    policy_loss_v = []
    value_loss_v = []
    entropy_v = []
    approx_kl_v = []
    early_stop_kl = 0

    seqs_per_batch = max(1, preset.minibatch_tokens // max(1, obs_t.shape[1]))

    for _ in range(preset.ppo_epochs):
        np.random.shuffle(idxs)
        stop_this_epoch = False

        for start in range(0, n_seq, seqs_per_batch):
            batch_idx = idxs[start:start + seqs_per_batch]

            b_obs = obs_t[batch_idx]
            b_act = act_t[batch_idx]
            b_old_logp = old_logp_t[batch_idx]
            b_adv = adv_t[batch_idx]
            b_ret = ret_t[batch_idx]
            b_old_val = old_val_t[batch_idx]
            b_mask = mask_t[batch_idx]

            hidden = unwrap_model(model).initial_state(batch_size=b_obs.shape[0], device=device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, values, _ = model(b_obs, hidden)
                values = values.squeeze(-1)

                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(b_act)
                entropy = dist.entropy()

                ratio = torch.exp(new_logp - b_old_logp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - preset.ppo_clip, 1.0 + preset.ppo_clip) * b_adv
                policy_loss = -torch.sum(torch.min(surr1, surr2) * b_mask) / (torch.sum(b_mask) + 1e-8)

                # value clipping
                value_pred_clipped = b_old_val + torch.clamp(values - b_old_val, -preset.value_clip, preset.value_clip)
                value_loss_unclipped = (values - b_ret) ** 2
                value_loss_clipped = (value_pred_clipped - b_ret) ** 2
                value_loss_all = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped)
                value_loss = torch.sum(value_loss_all * b_mask) / (torch.sum(b_mask) + 1e-8)

                entropy_mean = torch.sum(entropy * b_mask) / (torch.sum(b_mask) + 1e-8)
                loss = policy_loss + preset.value_coef * value_loss - ent_coef * entropy_mean

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), preset.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                approx_kl = torch.sum((b_old_logp - new_logp) * b_mask) / (torch.sum(b_mask) + 1e-8)

            total_loss_v.append(float(loss.item()))
            policy_loss_v.append(float(policy_loss.item()))
            value_loss_v.append(float(value_loss.item()))
            entropy_v.append(float(entropy_mean.item()))
            approx_kl_v.append(float(approx_kl.item()))

            if float(approx_kl.item()) > float(preset.target_kl):
                early_stop_kl += 1
                stop_this_epoch = True
                break

        if stop_this_epoch:
            break

    return {
        "loss_total": float(np.mean(total_loss_v)) if total_loss_v else 0.0,
        "loss_policy": float(np.mean(policy_loss_v)) if policy_loss_v else 0.0,
        "loss_value": float(np.mean(value_loss_v)) if value_loss_v else 0.0,
        "entropy": float(np.mean(entropy_v)) if entropy_v else 0.0,
        "approx_kl": float(np.mean(approx_kl_v)) if approx_kl_v else 0.0,
        "early_stop_kl": int(early_stop_kl),
    }


# ============================================================
# Checkpoint
# ============================================================

def ckpt_path(ckpt_dir: str, preset_name: str, seed: int):
    ensure_dir(ckpt_dir)
    return os.path.join(ckpt_dir, f"{preset_name}_seed{seed}_latest.pt")


def meta_path(ckpt_dir: str, preset_name: str, seed: int):
    ensure_dir(ckpt_dir)
    return os.path.join(ckpt_dir, f"{preset_name}_seed{seed}_meta.json")


def save_json(path: str, data: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_checkpoint(
    path: str,
    episode: int,
    global_step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    obs_rms: RunningMeanStd,
    raw_train_rewards: List[float],
    train_reward_history: List[float],
    eval_history: List[Dict[str, Any]],
    config: Dict[str, Any],
):
    payload = {
        "episode": int(episode),
        "global_step": int(global_step),
        "model": model_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "obs_rms": obs_rms.state_dict(),
        "raw_train_rewards": list(raw_train_rewards),
        "train_reward_history": list(train_reward_history),
        "eval_history": list(eval_history),
        "config": config,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    obs_rms: RunningMeanStd,
    device: torch.device,
):
    payload = torch.load(path, map_location=device)

    load_model_state_dict(model, payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])

    try:
        scaler.load_state_dict(payload["scaler"])
    except Exception:
        pass

    if "obs_rms" in payload:
        obs_rms.load_state_dict(payload["obs_rms"])

    return {
        "episode": int(payload.get("episode", -1)),
        "global_step": int(payload.get("global_step", 0)),
        "raw_train_rewards": list(payload.get("raw_train_rewards", [])),
        "train_reward_history": list(payload.get("train_reward_history", [])),
        "eval_history": list(payload.get("eval_history", [])),
        "config": payload.get("config", {}),
    }


# ============================================================
# One experiment
# ============================================================

def run_experiment(
    preset: EnvPreset,
    seed: int,
    total_episodes_override: Optional[int],
    eval_every: int,
    eval_episodes: int,
    save_every: int,
    log_every: int,
    device: torch.device,
    ckpt_dir: str,
    compile_models: bool,
    resume: bool,
    amp: bool,
):
    set_global_seed(seed)

    env = build_env(preset, seed=seed)
    shaper = get_shaper(preset.shaper)

    obs_dim = int(np.prod(env.observation_space.shape))
    assert isinstance(env.action_space, gym.spaces.Discrete), "Only discrete action spaces supported"
    action_dim = int(env.action_space.n)

    model = RecurrentActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        mlp_hidden=preset.mlp_hidden,
        lstm_hidden=preset.lstm_hidden,
    ).to(device)

    if compile_models:
        model = maybe_compile(model, True)

    optimizer = torch.optim.Adam(unwrap_model(model).parameters(), lr=preset.lr, eps=1e-5)
    use_amp = bool(amp) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    obs_rms = RunningMeanStd(shape=(obs_dim,))

    episodes_target = int(total_episodes_override) if total_episodes_override is not None else int(preset.recommended_episodes)

    start_episode = 0
    global_step = 0
    raw_train_rewards: List[float] = []
    train_reward_history: List[float] = []
    eval_history: List[Dict[str, Any]] = []

    cpath = ckpt_path(ckpt_dir, preset.name, seed)
    mpath = meta_path(ckpt_dir, preset.name, seed)

    if resume and os.path.exists(cpath):
        print(f"[resume] loading {cpath}")
        state = load_checkpoint(cpath, model, optimizer, scaler, obs_rms, device)
        start_episode = int(state["episode"]) + 1
        global_step = int(state["global_step"])
        raw_train_rewards = state["raw_train_rewards"]
        train_reward_history = state["train_reward_history"]
        eval_history = state["eval_history"]
        print(f"[resume] preset={preset.name} seed={seed} start_episode={start_episode}")

    buffer = RolloutBuffer()

    try:
        if start_episode >= episodes_target:
            print(f"[skip] already finished: {preset.name} seed={seed}")
            result = {
                "preset": preset.name,
                "env_id": preset.env_id,
                "seed": seed,
                "episodes_completed": len(raw_train_rewards),
                **compute_train_metrics(raw_train_rewards, preset.solve_score),
            }
            if eval_history:
                result.update(eval_history[-1])
            return result

        obs, _ = env.reset(seed=seed + 123)
        shaper.reset(obs)
        hidden = unwrap_model(model).initial_state(batch_size=1, device=device)

        episode_idx = start_episode
        ep_raw_reward = 0.0
        ep_train_reward = 0.0
        ep_steps = 0

        while episode_idx < episodes_target:
            progress01 = episode_idx / max(1, episodes_target - 1)
            env.set_progress(progress01)

            # lr annealing
            lr_now = linear_interp(preset.lr, preset.lr * preset.lr_end_scale, progress01)
            for g in optimizer.param_groups:
                g["lr"] = lr_now

            obs_rms.update(np.asarray(obs, dtype=np.float32)[None, :])
            obs_n = obs_rms.normalize(obs)

            obs_t = torch.as_tensor(obs_n, dtype=torch.float32, device=device).view(1, 1, -1)
            action, log_prob, value, _entropy, hidden = unwrap_model(model).act(obs_t, hidden)

            next_obs, raw_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            train_reward = shaper.train_reward(
                obs=obs,
                action=action,
                next_obs=next_obs,
                raw_reward=raw_reward,
                terminated=terminated,
                truncated=truncated,
            )

            buffer.add(
                obs=obs_n,
                action=action,
                train_reward=train_reward,
                raw_reward=raw_reward,
                done=done,
                log_prob=log_prob,
                value=value,
                episode_id=episode_idx,
            )

            obs = next_obs
            ep_raw_reward += float(raw_reward)
            ep_train_reward += float(train_reward)
            ep_steps += 1
            global_step += 1

            if done or ep_steps >= preset.max_steps:
                buffer.bootstrap_values.append(0.0)
                raw_train_rewards.append(float(ep_raw_reward))
                train_reward_history.append(float(ep_train_reward))

                if (episode_idx % log_every) == 0:
                    recent = raw_train_rewards[-20:] if len(raw_train_rewards) >= 20 else raw_train_rewards
                    recent_mean = float(np.mean(recent)) if recent else 0.0
                    flicker_now, noise_now = env.get_current_params()
                    print(
                        f"[{preset.name} seed={seed}] "
                        f"ep={episode_idx:05d} "
                        f"rawR={ep_raw_reward:8.3f} "
                        f"trainR={ep_train_reward:8.3f} "
                        f"recent20={recent_mean:8.3f} "
                        f"flicker={flicker_now:.3f} "
                        f"noise={noise_now:.4f} "
                        f"lr={lr_now:.6f} "
                        f"global_step={global_step}"
                    )

                if ((episode_idx + 1) % eval_every) == 0 or (episode_idx + 1) == episodes_target:
                    eval_result = evaluate_policy(
                        model=model,
                        preset=preset,
                        device=device,
                        obs_rms=obs_rms,
                        n_episodes=eval_episodes,
                        greedy=True,
                    )
                    eval_result["eval_at_episode"] = int(episode_idx + 1)
                    eval_history.append(eval_result)

                    print(
                        f"[eval] {preset.name} seed={seed} "
                        f"ep={episode_idx+1} "
                        f"mean={eval_result['eval_mean']:.3f} "
                        f"std={eval_result['eval_std']:.3f}"
                    )

                if ((episode_idx + 1) % save_every) == 0 or STOP_REQUESTED:
                    save_checkpoint(
                        path=cpath,
                        episode=episode_idx,
                        global_step=global_step,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        obs_rms=obs_rms,
                        raw_train_rewards=raw_train_rewards,
                        train_reward_history=train_reward_history,
                        eval_history=eval_history,
                        config={
                            "preset": asdict(preset),
                            "seed": seed,
                            "episodes_target": episodes_target,
                        },
                    )
                    save_json(mpath, {
                        "preset": preset.name,
                        "env_id": preset.env_id,
                        "seed": seed,
                        "last_saved_episode": episode_idx,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "global_step": global_step,
                        "status": "running_or_interrupted",
                    })
                    print(f"[ckpt] saved {cpath}")

                episode_idx += 1

                if STOP_REQUESTED:
                    print("[stop] graceful stop requested.")
                    break

                obs, _ = env.reset(seed=seed * 100000 + episode_idx)
                shaper.reset(obs)
                hidden = unwrap_model(model).initial_state(batch_size=1, device=device)
                ep_raw_reward = 0.0
                ep_train_reward = 0.0
                ep_steps = 0

            # update
            if len(buffer) >= preset.steps_per_update or (STOP_REQUESTED and len(buffer) > 0):
                needed_segments = len(split_into_segments_preview(buffer))
                if len(buffer.bootstrap_values) < needed_segments:
                    with torch.no_grad():
                        obs_n = obs_rms.normalize(obs)
                        obs_t = torch.as_tensor(obs_n, dtype=torch.float32, device=device).view(1, 1, -1)
                        _, value_t, _ = model(obs_t, hidden)
                        bootstrap_value = float(value_t[:, -1, :].item()) if not done else 0.0
                    buffer.bootstrap_values.append(bootstrap_value)

                update_stats = ppo_update(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    buffer=buffer,
                    preset=preset,
                    device=device,
                    use_amp=use_amp,
                    progress01=progress01,
                )

                print(
                    f"[update] {preset.name} seed={seed} "
                    f"loss={update_stats['loss_total']:.4f} "
                    f"policy={update_stats['loss_policy']:.4f} "
                    f"value={update_stats['loss_value']:.4f} "
                    f"entropy={update_stats['entropy']:.4f} "
                    f"kl={update_stats['approx_kl']:.4f} "
                    f"early_stop_kl={update_stats['early_stop_kl']}"
                )

                buffer.clear()

                if STOP_REQUESTED:
                    break

        final_episode = min(max(start_episode, len(raw_train_rewards)) - 1, episodes_target - 1)
        if final_episode >= 0:
            save_checkpoint(
                path=cpath,
                episode=final_episode,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                obs_rms=obs_rms,
                raw_train_rewards=raw_train_rewards,
                train_reward_history=train_reward_history,
                eval_history=eval_history,
                config={
                    "preset": asdict(preset),
                    "seed": seed,
                    "episodes_target": episodes_target,
                },
            )
            save_json(mpath, {
                "preset": preset.name,
                "env_id": preset.env_id,
                "seed": seed,
                "last_saved_episode": final_episode,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "global_step": global_step,
                "status": "finished_or_stopped",
            })
            print(f"[ckpt] final saved {cpath}")

        result = {
            "preset": preset.name,
            "env_id": preset.env_id,
            "seed": seed,
            "episodes_completed": len(raw_train_rewards),
            **compute_train_metrics(raw_train_rewards, preset.solve_score),
        }
        if eval_history:
            result.update(eval_history[-1])

        return result

    finally:
        try:
            env.close()
        except Exception:
            pass
        cleanup_torch(device)


# ============================================================
# Aggregation / save
# ============================================================

def aggregate_results(results: List[Dict[str, Any]]):
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        grouped.setdefault(str(r["preset"]), []).append(r)

    keys = [
        "auc_mean",
        "last100_mean",
        "last100_std",
        "best_reward",
        "first_solve_ep",
        "eval_mean",
        "eval_std",
        "episodes_completed",
    ]

    out = []
    for preset_name, rows in grouped.items():
        item = {
            "preset": preset_name,
            "n_seeds": len(rows),
        }
        for k in keys:
            vals = [float(x[k]) for x in rows if k in x]
            item[k] = float(np.mean(vals)) if vals else 0.0
        out.append(item)

    out.sort(key=lambda x: x["last100_mean"], reverse=True)
    return out


def save_csv(path: str, rows: List[Dict[str, Any]]):
    ensure_dir(os.path.dirname(path) or ".")
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return

    cols = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def save_json_rows(path: str, rows: List[Dict[str, Any]]):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


# ============================================================
# Args
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(add_help=True)

    p.add_argument("--device", type=str, default=None, help="cuda:0 / cpu / auto(None)")
    p.add_argument("--amp", action="store_true", help="use amp on cuda")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.set_defaults(amp=USE_AMP_DEFAULT)

    p.add_argument("--compile", action="store_true")
    p.add_argument("--resume", action="store_true")

    p.add_argument(
        "--presets",
        type=str,
        default="cartpole_reliable,acrobot_reliable,mountaincar_reliable",
        help="comma-separated preset names",
    )
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--total-episodes", type=int, default=None)

    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=20)

    p.add_argument("--ckpt-dir", type=str, default="./checkpoints_robust_recurrent_ppo")
    p.add_argument("--raw-csv", type=str, default="./robust_recurrent_ppo_raw.csv")
    p.add_argument("--summary-csv", type=str, default="./robust_recurrent_ppo_summary.csv")
    p.add_argument("--raw-json", type=str, default="./robust_recurrent_ppo_raw.json")
    p.add_argument("--summary-json", type=str, default="./robust_recurrent_ppo_summary.json")

    # 노트북 끝나면 안전 종료
    p.add_argument("--shutdown-kernel-on-finish", action="store_true")

    # notebook 안전
    args, _unknown = p.parse_known_args()
    return args


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    presets_map = get_presets()

    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    preset_names = [x.strip() for x in args.presets.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    selected_presets = []
    for name in preset_names:
        if name not in presets_map:
            raise ValueError(f"unknown preset: {name}. available={list(presets_map.keys())}")
        selected_presets.append(presets_map[name])

    print("============================================================")
    print("device       =", device)
    print("amp          =", bool(args.amp and device.type == "cuda"))
    print("compile      =", args.compile)
    print("presets      =", [p.name for p in selected_presets])
    print("seeds        =", seeds)
    print("episodes     =", args.total_episodes, "(None이면 preset recommended_episodes)")
    print("============================================================")

    raw_results = []
    started = time.time()

    try:
        for preset in selected_presets:
            for seed in seeds:
                if STOP_REQUESTED:
                    break

                print(f"\n===== RUN preset={preset.name} seed={seed} =====")
                result = run_experiment(
                    preset=preset,
                    seed=seed,
                    total_episodes_override=args.total_episodes,
                    eval_every=int(args.eval_every),
                    eval_episodes=int(args.eval_episodes),
                    save_every=int(args.save_every),
                    log_every=int(args.log_every),
                    device=device,
                    ckpt_dir=args.ckpt_dir,
                    compile_models=args.compile,
                    resume=args.resume,
                    amp=args.amp,
                )
                raw_results.append(result)

            if STOP_REQUESTED:
                break

    finally:
        summary = aggregate_results(raw_results)

        save_csv(args.raw_csv, raw_results)
        save_csv(args.summary_csv, summary)
        save_json_rows(args.raw_json, raw_results)
        save_json_rows(args.summary_json, summary)

        elapsed = time.time() - started

        print("\n================ FINAL SUMMARY ================")
        if summary:
            cols = [
                "preset", "n_seeds", "episodes_completed",
                "auc_mean", "last100_mean", "best_reward",
                "first_solve_ep", "eval_mean", "eval_std"
            ]
            widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in summary)) for c in cols}
            header = " | ".join(c.ljust(widths[c]) for c in cols)
            sep = "-+-".join("-" * widths[c] for c in cols)
            print(header)
            print(sep)

            for r in summary:
                pr = dict(r)
                for k in ["episodes_completed", "n_seeds", "first_solve_ep"]:
                    if k in pr:
                        pr[k] = int(float(pr[k]))
                for k in ["auc_mean", "last100_mean", "best_reward", "eval_mean", "eval_std"]:
                    if k in pr:
                        pr[k] = f"{float(pr[k]):.3f}"
                print(" | ".join(str(pr.get(c, "")).ljust(widths[c]) for c in cols))
        else:
            print("(no results)")

        print("\nraw csv     :", args.raw_csv)
        print("summary csv :", args.summary_csv)
        print("raw json    :", args.raw_json)
        print("summary json:", args.summary_json)
        print(f"elapsed_sec : {elapsed:.2f}")

        print("\n[recommended run]")
        print(
            "python robust_multi_env_recurrent_ppo.py "
            "--device cuda:0 "
            "--presets cartpole_reliable,acrobot_reliable,mountaincar_reliable "
            "--seeds 0,1,2 "
            "--resume"
        )

        print("\n[more POMDP]")
        print(
            "python robust_multi_env_recurrent_ppo.py "
            "--device cuda:0 "
            "--presets cartpole_medium,acrobot_medium,mountaincar_medium "
            "--seeds 0,1,2 "
            "--resume"
        )

        print("\n[hard partial test]")
        print(
            "python robust_multi_env_recurrent_ppo.py "
            "--device cuda:0 "
            "--presets cartpole_partial_hard "
            "--seeds 0,1,2 "
            "--resume"
        )

        print("\n[important note]")
        print(
            "이 코드는 학습 안정성을 위해 reward shaping(train only)과 "
            "POMDP severity curriculum을 사용한다. "
            "평가/summary는 raw reward 기준으로 기록된다."
        )

        cleanup_torch(device)
        maybe_shutdown_kernel(bool(args.shutdown_kernel_on_finish))


if __name__ == "__main__":
    main()


# In[10]:


get_ipython().system('ls -lh ./checkpoints_robust_recurrent_ppo')


# In[17]:


get_ipython().system('python robust_multi_env_recurrent_ppo.py --device cuda:0 --resume')


# In[18]:


get_ipython().system('pwd')


# In[19]:


get_ipython().system('ls')


# In[20]:


get_ipython().system('ls /workspace/soyoung/pomdp')


# In[ ]:




