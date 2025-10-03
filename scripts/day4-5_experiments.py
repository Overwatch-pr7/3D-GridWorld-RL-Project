# scripts/exp_day4_5.py
from __future__ import annotations
import os, csv, time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from src.gridworld import GridSpec, Gridworld3D
from src.qlearning import QLearningAgent, QConfig, EpsilonSchedule, evaluate_random_policy
from src.utils import plot_learning_curve, ensure_dir
from src.vis import plot_slice

RESULTS_DIR = "results"
CURVES_DIR  = os.path.join(RESULTS_DIR, "learning_curves")
POL_DIR     = os.path.join(RESULTS_DIR, "policies")
HEAT_DIR    = os.path.join(RESULTS_DIR, "value_heatmaps")
TAB_DIR     = os.path.join(RESULTS_DIR, "tables")

def run_one(spec: GridSpec, qcfg: QConfig, tag: str):
    """Train one config, save curve & artifacts, return summary row."""
    env = Gridworld3D(spec)
    agent = QLearningAgent(env, qcfg)

    t0 = time.time()
    out = agent.train()
    dt = time.time() - t0
    returns = out["returns"]

    # Save curve
    ensure_dir(CURVES_DIR)
    ensure_dir(POL_DIR)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    curve_path = os.path.join(CURVES_DIR, f"{tag}_{ts}.png")
    plot_learning_curve(returns, curve_path, ma_k=50, title=f"Q-learning: {tag}")

    # Evaluate greedy vs random
    avg_greedy = agent.evaluate_policy(episodes=100)
    avg_random = evaluate_random_policy(env, episodes=100, seed=777)

    # Save policy/Q
    np.save(os.path.join(POL_DIR, f"{tag}_policy.npy"), agent.policy_matrix())
    np.save(os.path.join(POL_DIR, f"{tag}_Q.npy"), agent.Q)

    # Heatmaps for z slices (0, mid, last)
    ensure_dir(HEAT_DIR)
    z_slices = [0, spec.D//2, spec.D-1]
    for z in z_slices:
        plot_slice(env, agent.Q, z, os.path.join(HEAT_DIR, f"{tag}_z{z}.png"),
                   title=f"{tag} (z={z})")

    final_ma = float(np.mean(returns[-100:])) if len(returns) >= 100 else float(np.mean(returns))
    return {
        "tag": tag,
        "episodes": qcfg.episodes,
        "alpha": qcfg.alpha,
        "gamma": qcfg.gamma,
        "p_intended": spec.p_intended,
        "step_cost": spec.step_cost,
        "obstacle_ratio": spec.obstacle_ratio,
        "avg_return_last100": final_ma,
        "greedy_avg": avg_greedy,
        "random_avg": avg_random,
        "train_seconds": dt,
        "curve_path": curve_path
    }

def bar_plot(results, attr: str, title: str, save_path: str):
    """Simple bar plot of greedy_avg grouped by tag order."""
    ensure_dir(os.path.dirname(save_path))
    tags = [r["tag"] for r in results]
    vals = [r["greedy_avg"] for r in results]
    plt.figure()
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(tags)), tags, rotation=30, ha='right')
    plt.ylabel("Greedy policy avg return")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def write_csv(rows, save_path: str):
    ensure_dir(os.path.dirname(save_path))
    if not rows: return
    keys = list(rows[0].keys())
    with open(save_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def main():
    ensure_dir(RESULTS_DIR)
    ensure_dir(TAB_DIR)

    # --- Base spec (assignment defaults) ---
    base_spec = GridSpec(
        H=6, W=6, D=6,
        p_intended=0.8,
        step_cost=-1.0,
        gamma=0.95,
        goal=(5,5,5),
        pit=(2,2,2),
        obstacle_ratio=0.12,
        start=(0,0,0),
        seed=12345
    )

    base_q = QConfig(
        alpha=0.2,
        gamma=base_spec.gamma,
        episodes=5000,
        eps_schedule=EpsilonSchedule(start=1.0, end=0.05, decay=0.999),
        seed=2024
    )

    all_rows = []

    # ---- Sweep 1: gamma ----
    gamma_vals = [0.90, 0.95, 0.99]
    rows_g = []
    for g in gamma_vals:
        spec = base_spec
        qcfg = QConfig(alpha=base_q.alpha, gamma=g, episodes=base_q.episodes,
                       eps_schedule=base_q.eps_schedule, seed=base_q.seed)
        tag = f"gamma_{g:.2f}"
        rows_g.append(run_one(spec, qcfg, tag))
    write_csv(rows_g, os.path.join(TAB_DIR, "sweep_gamma.csv"))
    bar_plot(rows_g, "greedy_avg", "Greedy return by gamma", os.path.join(RESULTS_DIR, "gamma_bar.png"))
    all_rows.extend(rows_g)

    # ---- Sweep 2: slip probability p_intended ----
    p_vals = [0.60, 0.80]
    rows_p = []
    for p in p_vals:
        spec = GridSpec(**{**base_spec.__dict__, "p_intended": p})
        qcfg = base_q
        tag = f"p_{p:.2f}"
        rows_p.append(run_one(spec, qcfg, tag))
    write_csv(rows_p, os.path.join(TAB_DIR, "sweep_p.csv"))
    bar_plot(rows_p, "greedy_avg", "Greedy return by slip p", os.path.join(RESULTS_DIR, "p_bar.png"))
    all_rows.extend(rows_p)

    # ---- Sweep 3: step cost ----
    step_vals = [-1.0, -0.5]
    rows_c = []
    for c in step_vals:
        spec = GridSpec(**{**base_spec.__dict__, "step_cost": c})
        qcfg = base_q
        tag = f"step_{c:.2f}".replace("-", "m")
        rows_c.append(run_one(spec, qcfg, tag))
    write_csv(rows_c, os.path.join(TAB_DIR, "sweep_step_cost.csv"))
    bar_plot(rows_c, "greedy_avg", "Greedy return by step cost", os.path.join(RESULTS_DIR, "step_bar.png"))
    all_rows.extend(rows_c)

    # ---- Combined log ----
    write_csv(all_rows, os.path.join(TAB_DIR, "all_results.csv"))
    print("\nSaved:")
    print(" - Curves in results/learning_curves/")
    print(" - Heatmaps in results/value_heatmaps/")
    print(" - Policies & Q in results/policies/")
    print(" - CSV tables in results/tables/")
    print(" - Comparison bars in results/")

if __name__ == "__main__":
    main()
