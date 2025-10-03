# src/qlearning.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Callable, Dict, Any

from .gridworld import Gridworld3D, GridSpec

@dataclass
class EpsilonSchedule:
    start: float = 1.0
    end: float = 0.05
    decay: float = 0.999  # multiplicative per step

    def update(self, eps: float) -> float:
        return max(self.end, eps * self.decay)

@dataclass
class QConfig:
    alpha: float = 0.2
    gamma: float = 0.95
    episodes: int = 5000
    max_steps_per_ep: Optional[int] = None  # default set in agent using grid size
    eps_schedule: EpsilonSchedule = field(default_factory=EpsilonSchedule) #corrected
    seed: Optional[int] = 123  # for action tie-breaking & any numpy ops in agent

class QLearningAgent:
    """
    Tabular Q-learning with ε-greedy exploration.
    Compatible with Gridworld3D (discrete states/actions).
    """
    def __init__(self, env: Gridworld3D, cfg: QConfig):
        self.env = env
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.nS, self.nA = env.nS, env.nA
        self.Q = np.zeros((self.nS, self.nA), dtype=np.float64)

        # default horizon: a few times grid volume to avoid infinitely long episodes
        if cfg.max_steps_per_ep is None:
            vol = env.H * env.W * env.D
            self.max_steps = int(3 * vol)
        else:
            self.max_steps = cfg.max_steps_per_ep

    # ---------- policies ----------
    def greedy_action(self, s_idx: int) -> int:
        # break ties randomly but reproducibly
        row = self.Q[s_idx]
        max_val = row.max()
        candidates = np.flatnonzero(row == max_val)
        return int(self.rng.choice(candidates))

    def epsilon_greedy(self, s_idx: int, eps: float) -> int:
        if self.rng.random() < eps:
            return int(self.rng.integers(self.nA))
        return self.greedy_action(s_idx)

    # ---------- learning ----------
    def train(self, progress_cb: Optional[Callable[[int, float, float], None]] = None) -> Dict[str, Any]:
        alpha, gamma = self.cfg.alpha, self.cfg.gamma
        eps = self.cfg.eps_schedule.start
        returns = np.zeros(self.cfg.episodes, dtype=np.float64)

        for ep in range(self.cfg.episodes):
            s = self.env.reset()
            G = 0.0
            steps = 0
            done = False

            while not done and steps < self.max_steps:
                a = self.epsilon_greedy(s, eps)
                s2, r, done, _ = self.env.step(a)

                # Q-learning TD target
                best_next = 0.0 if done else np.max(self.Q[s2])
                td_target = r + gamma * best_next
                td_error = td_target - self.Q[s, a]
                self.Q[s, a] += alpha * td_error

                s = s2
                G += r
                steps += 1

            eps = self.cfg.eps_schedule.update(eps)
            returns[ep] = G

            if progress_cb and ((ep + 1) % 100 == 0 or ep == 0):
                progress_cb(ep + 1, G, eps)

        return {"Q": self.Q, "returns": returns}

    # ---------- evaluation ----------
    def policy_matrix(self) -> np.ndarray:
        return np.argmax(self.Q, axis=1)

    def evaluate_policy(self, episodes: int = 100) -> float:
        """Evaluate greedy policy with no exploration; returns avg return."""
        total = 0.0
        for _ in range(episodes):
            s = self.env.reset()
            done = False
            steps = 0
            G = 0.0
            while not done and steps < self.max_steps:
                a = self.greedy_action(s)
                s, r, done, _ = self.env.step(a)
                G += r
                steps += 1
            total += G
        return total / episodes

def evaluate_random_policy(env: Gridworld3D, episodes: int = 100, seed: Optional[int] = 999) -> float:
    rng = np.random.default_rng(seed)
    vol = env.H * env.W * env.D
    max_steps = int(3 * vol)
    total = 0.0
    for _ in range(episodes):
        s = env.reset()
        done = False
        steps = 0
        G = 0.0
        while not done and steps < max_steps:
            a = int(rng.integers(env.nA))
            s, r, done, _ = env.step(a)
            G += r
            steps += 1
        total += G
    return total / episodes
