Q-Learning in a 3D Gridworld

Tabular Q-learning on a stochastic 3D Gridworld with slip dynamics, step cost, and absorbing terminals. Includes a clean Python package, training/evaluation scripts, experiment sweeps, inline visualizations (value heatmaps + policy arrows), and a Colab/Jupyter notebook version.

Features

Environment: 3D Gridworld (default 6×6×6), six actions (±x, ±y, ±z), stochastic slip: intended move with prob p, otherwise uniform among four perpendicular moves. Blocked moves → stay put. Absorbing terminals (goal +50, pit −50).

Agent: Tabular Q-learning with ε-greedy exploration and tie-breaks handled stochastically.

Experiments: One-factor sweeps over discount γ, slip probability p, and step cost; learning curves, greedy vs random evaluation, and value/policy visualizations for ≥3 z-slices.

Reproducibility: Fixed RNG seeds for obstacle placement and agent behavior.

Notebook: Self-contained .ipynb in experiments/ for Colab/Jupyter.

Repo Structure
3D-GridWorld-RL-Project/
│
├─ src/
│  ├─ gridworld.py      # Environment (GridSpec, Gridworld3D)
│  ├─ qlearning.py      # QLearningAgent, QConfig, EpsilonSchedule, evaluate_random_policy
│  ├─ utils.py          # plotting helpers (learning curve, moving average)
│  └─ vis.py            # value heatmaps + greedy policy arrows (per z-slice)
│
├─ scripts/
│  ├─ day3_training.py  # baseline training + curve + eval
│  └─ exp_day4_5.py     # sweeps: gamma / slip p / step cost; saves plots & CSVs
│
├─ tests/
│  └─ day1.py           # environment sanity tests (slip distribution, terminals, boundaries)
│
├─ results/
│  ├─ learning_curves/  # saved learning curves (PNG)
│  ├─ value_heatmaps/   # value heatmaps + arrows for z-slices (PNG)
│  ├─ policies/         # saved NumPy arrays: policy.npy, Q.npy
│  └─ tables/           # CSV summaries for sweeps
│
├─ experiments/
│  └─ 3d_gridworld_qlearning.ipynb   # Colab/Jupyter notebook version (revised)
│
├─ requirements.txt
└─ README.md


Make sure __init__.py exists (can be empty) in src/, scripts/, and tests/ so module imports work when running with python -m.

Getting Started
Prerequisites

Python 3.9–3.12

Recommended: a virtual environment (venv or conda)

pip install -r requirements.txt


requirements.txt should minimally include:

numpy
matplotlib

Quick Start (CLI)

Run all commands from the repo root (the folder that contains src/ and scripts/).

1) Sanity tests (environment)
python -m tests.day1


You should see: All Day 1–2 environment tests passed ✅

2) Baseline training (Day 3)
python -m scripts.day3_training


Outputs:

Learning curve PNG → results/learning_curves/…png

Saved policy/Q arrays → results/policies/day3_policy.npy, results/policies/day3_Q.npy

Console shows greedy vs random average return (100 episodes)

3) Experiments & visualizations (Day 4–5)
python -m scripts.exp_day4_5


Outputs:

Curves per run → results/learning_curves/*.png

Heatmaps (z = 0, mid, last) → results/value_heatmaps/*.png

Saved policies/Q tables → results/policies/*_policy.npy, *_Q.npy

Sweep summary CSVs → results/tables/sweep_gamma.csv, sweep_p.csv, sweep_step_cost.csv, all_results.csv

Comparison bars → results/gamma_bar.png, results/p_bar.png, results/step_bar.png

Notebook Usage (Colab/Jupyter)

A single-file, self-contained notebook is provided at:

experiments/3d_gridworld_qlearning.ipynb


How to run (Colab):

Upload the repo or open the notebook directly in Google Colab.

Run the cells top to bottom:

Env cell (defines GridSpec, Gridworld3D)

Agent cell (defines QLearningAgent, QConfig, EpsilonSchedule, evaluate_random_policy)

Train + plot (prints greedy vs random)

Heatmap helper

Sweeps cell (runs γ / p / step-cost experiments, plots curves + heatmaps, prints CSV text)

If Colab restarts, re-run the Env + Agent cells before the training/sweeps cells.

Configuration

Key knobs live in two dataclasses:

Environment: src/gridworld.py → GridSpec

H, W, D (default 6 each), p_intended (slip prob for intended move), step_cost (−1), gamma (0.95), goal, pit, obstacle_ratio (~0.12), start, seed (reproducible world)

Agent: src/qlearning.py → QConfig

alpha (learning rate), gamma, episodes, max_steps_per_ep (default ≈ 3×volume), eps_schedule (start, end, decay), seed (agent RNG / tie-breaks)

What to Expect (Default Settings)

Learning curve moving average stabilizes around +30 (± a bit) on 6×6×6 with p_intended=0.8, step_cost=-1, gamma=0.95.

Greedy policy average return ≫ random baseline (random often ~ −200 to −300).

Heatmaps show bright corridors toward the goal; obstacles appear as gaps; arrows (xy) bend around obstacles; z± actions are shown without arrows (zero xy displacement).

Reproducibility

Obstacle placement and agent randomness use fixed seeds (GridSpec.seed, QConfig.seed).

Terminals are absorbing: if you are on a terminal and call step, you immediately receive ±50 and done=True. Entering a terminal also terminates.

Troubleshooting

ModuleNotFoundError: No module named 'src'
Run from the repo root, not inside scripts/ or tests/, and use python -m scripts.... Ensure src/__init__.py exists.

Dataclass mutable default error (Python 3.12):
In QConfig, eps_schedule must use field(default_factory=EpsilonSchedule).

Matplotlib backend issues (some Windows setups):
Add plt.switch_backend("Agg") at the top of plotting scripts to save files without showing windows.

Citing & Related Work

If you include a related-work section in your report, a concise, relevant citation is:

Ergon Cugler de Moraes Silva, “From Two-Dimensional to Three-Dimensional Environment with Q-Learning: Modeling Autonomous Navigation with Reinforcement Learning and no Libraries,” arXiv:2403.18219 (2024).

(Plus any course notes or textbooks you referenced.)

License

Add your preferred license (e.g., MIT) in LICENSE and mention it here. If unsure, MIT is common for course projects.

Acknowledgments

Thanks to teammates/instructors for feedback, and to open-source contributors for NumPy/Matplotlib.

Happy experimenting! If you want, I can also add a tiny make_submission_zip.py script that packs your code + selected results + report into a single archive for upload.
