# scripts/train_day3.py
from __future__ import annotations
import os
from datetime import datetime

from src.gridworld import GridSpec, Gridworld3D
from src.qlearning import QLearningAgent, QConfig, EpsilonSchedule, evaluate_random_policy
from src.utils import plot_learning_curve, ensure_dir

def main():
    # ----- 1) Environment spec (matches your assignment defaults) -----
    spec = GridSpec(
        H=6, W=6, D=6,
        p_intended=0.8,
        step_cost=-1.0,
        gamma=0.95,
        goal=(5,5,5),
        pit=(2,2,2),
        obstacle_ratio=0.12,   # ~12% obstacles
        start=(0,0,0),
        seed=12345             # reproducible world
    )
    env = Gridworld3D(spec)

    # ----- 2) Q-learning hyperparameters -----
    qcfg = QConfig(
        alpha=0.2,
        gamma=spec.gamma,      # match env gamma
        episodes=5000,
        max_steps_per_ep=None, # default = 3 * (H*W*D)
        eps_schedule=EpsilonSchedule(start=1.0, end=0.05, decay=0.999),
        seed=2024
    )
    agent = QLearningAgent(env, qcfg)

    # ----- 3) Train -----
    def progress(ep, G, eps):
        # lightweight console log
        if ep % 500 == 0 or ep in (1,):
            print(f"[ep={ep:5d}] return={G:7.2f}  eps={eps:.3f}")

    out = agent.train(progress_cb=progress)
    returns = out["returns"]

    # ----- 4) Save learning curve -----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    curve_path = os.path.join("results", "learning_curves", f"day3_curve_{timestamp}.png")
    plot_learning_curve(returns, curve_path, ma_k=50, title="Q-learning on 3D Gridworld")
    print(f"Saved learning curve to: {curve_path}")

    # ----- 5) Evaluate greedy policy vs random baseline -----
    avg_greedy = agent.evaluate_policy(episodes=100)
    avg_random = evaluate_random_policy(env, episodes=100, seed=777)
    print("\n=== Evaluation (100 episodes, no exploration) ===")
    print(f"Greedy policy average return : {avg_greedy:7.3f}")
    print(f"Random policy average return : {avg_random:7.3f}")

    # Optionally, save the policy matrix & Q for later visualization/analysis
    ensure_dir("results/policies")
    import numpy as np
    policy = agent.policy_matrix()
    np.save("results/policies/day3_policy.npy", policy)
    np.save("results/policies/day3_Q.npy", agent.Q)
    print("Saved policy and Q-table under results/policies/")

if __name__ == "__main__":
    main()
