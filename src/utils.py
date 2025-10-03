# src/utils.py
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def moving_average(x: np.ndarray, k: int = 50) -> np.ndarray:
    if k <= 1:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = (c[k:] - c[:-k]) / float(k)
    # pad head so arrays align visually
    pad = np.full(k - 1, out[0])
    return np.concatenate([pad, out])

def plot_learning_curve(returns: np.ndarray, save_path: str, ma_k: int = 50, title: str = "") -> None:
    ensure_dir(os.path.dirname(save_path))
    plt.figure()
    plt.plot(returns, label="Return per episode")
    if ma_k > 1:
        plt.plot(moving_average(returns, ma_k), linewidth=2, label=f"Moving avg (k={ma_k})")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(title or "Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
