# src/vis.py
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Tuple
from .utils import ensure_dir

# map action index -> 2D vector per z-slice (x,y plane arrows)
ARROW_2D = {
    0: ( +1,  0),  # +x
    1: ( -1,  0),  # -x
    2: (  0, +1),  # +y
    3: (  0, -1),  # -y
    4: (  0,  0),  # +z (no xy displacement)
    5: (  0,  0),  # -z (no xy displacement)
}

def values_from_Q(Q: np.ndarray) -> np.ndarray:
    """V(s) = max_a Q(s,a) as a 1D array (len = nS)."""
    return np.max(Q, axis=1)

def reshape_slice(env, V_1d, z: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return V_2d (H x W) and integer action grid (H x W) for fixed z.
    Only free cells appear in env.idx2state; we fill obstacles with NaN.
    """
    H, W, D = env.H, env.W, env.D
    V2 = np.full((H, W), np.nan, dtype=float)
    A2 = np.full((H, W), -1, dtype=int)  # greedy action index per cell
    for s_idx, (x,y,zz) in enumerate(env.idx2state):
        if zz != z: continue
        V2[x, y] = V_1d[s_idx]
        # greedy action
        a = int(np.argmax(env._agent_Q[s_idx])) if hasattr(env, "_agent_Q") else -1
        A2[x, y] = a
    # meshgrid for quiver
    X, Y = np.meshgrid(range(W), range(H))
    return V2, A2, (X, Y)

def plot_slice(env, Q, z: int, save_path: str, title: str = ""):
    ensure_dir(os.path.dirname(save_path))
    V = values_from_Q(Q)
    env._agent_Q = Q  # hack so reshape_slice can fetch greedy action
    V2, A2, (X, Y) = reshape_slice(env, V, z)

    U = np.zeros_like(V2, dtype=float)  # x-component
    Wv = np.zeros_like(V2, dtype=float) # y-component
    for x in range(env.H):
        for y in range(env.W):
            a = A2[x, y]
            if a < 0: continue
            dx, dy = ARROW_2D.get(a, (0,0))
            U[x, y] = dx
            Wv[x, y] = dy

    plt.figure()
    # Heatmap of values
    plt.imshow(np.flipud(V2.T), interpolation="nearest")  # transpose & flip for nicer orientation
    # Quiver (arrows) — only for xy actions; z± shows no arrow (dx=dy=0)
    # We also flip Y for display parity
    mask = ~np.isnan(V2)
    xs = X[mask]; ys = Y[mask]
    us = U[mask]; vs = Wv[mask]
    plt.quiver(xs, env.H-1-ys, us, -vs, angles='xy', scale_units='xy', scale=1)

    plt.colorbar(label="V(s) = max_a Q(s,a)")
    plt.title(title or f"Value & policy, z={z}")
    plt.xlabel("y (cols)"); plt.ylabel("x (rows)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    delattr(env, "_agent_Q")
