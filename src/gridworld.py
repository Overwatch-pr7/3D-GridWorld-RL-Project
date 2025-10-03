# src/gridworld.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional

Coord = Tuple[int, int, int]

@dataclass
class GridSpec:
    H: int = 6
    W: int = 6
    D: int = 6
    p_intended: float = 0.8     # slip parameter: P(do intended move)
    step_cost: float = -1.0
    gamma: float = 0.95
    goal: Coord = (5, 5, 5)
    pit: Coord = (2, 2, 2)
    obstacle_ratio: float = 0.12  # ~12% by default, inside 10–15% band
    start: Coord = (0, 0, 0)
    seed: Optional[int] = 12345  # fix for reproducibility

class Gridworld3D:
    """
    - States: free cells (x,y,z) only
    - Actions: 6 unit moves (±x, ±y, ±z)
    - Transitions: with prob p do intended; with prob (1-p) slip uniformly
      to one of four perpendicular actions. If blocked, stay.
    - Rewards: step cost each move; +50 at goal (absorbing), -50 at pit (absorbing)
    - Discount gamma stored for completeness (used by learners)
    """
    # displacement vectors
    ACTIONS = {
        0: np.array([+1,  0,  0]),  # +x (East)
        1: np.array([-1,  0,  0]),  # -x (West)
        2: np.array([ 0, +1,  0]),  # +y (North)
        3: np.array([ 0, -1,  0]),  # -y (South)
        4: np.array([ 0,  0, +1]),  # +z (Up)
        5: np.array([ 0,  0, -1]),  # -z (Down)
    }

    def __init__(self, spec: GridSpec):
        self.spec = spec
        self.rng = np.random.default_rng(spec.seed)
        self.H, self.W, self.D = spec.H, spec.W, spec.D
        self.gamma = spec.gamma
        # Place obstacles reproducibly, then ensure start/terminals are clear and reachable.
        self.obstacles: Set[Coord] = self._generate_obstacles(spec.obstacle_ratio)
        # Clear mandatory cells
        for forced_free in (spec.start, spec.goal, spec.pit):
            if forced_free in self.obstacles:
                self.obstacles.remove(forced_free)
        # Terminals
        self.terminals: Dict[Coord, float] = {
            spec.goal: +50.0,
            spec.pit: -50.0
        }
        # Build index mapping over FREE (non-obstacle) cells
        self._build_state_index()
        # Validate start exists as free cell; if not, pick first free
        self.start: Coord = spec.start if spec.start in self.state2idx else self.idx2state[0]
        self._s: Coord = self.start  # current coord

    # ----- Public API -----
    @property
    def nS(self) -> int:
        """Number of free states (excluding obstacles)."""
        return len(self.idx2state)

    @property
    def nA(self) -> int:
        """Number of actions (6)."""
        return 6

    def reset(self) -> int:
        """Reset episode to start coordinate; return state index."""
        self._s = self.start
        return self.state2idx[self._s]

    def step(self, a: int) -> Tuple[int, float, bool, dict]:
        # Absorbing terminals: if already at a terminal, stay and finish
        if self._s in self.terminals:
            r = self.terminals[self._s]
            return self.state2idx[self._s], r, True, {}
        """
        Stochastic step with slip:
        - choose intended action with prob p
        - otherwise pick uniformly among 4 perpendicular actions
        - if chosen move is blocked (out-of-bounds or obstacle), agent stays
        Returns (s_idx_next, reward, done, info)
        """
        assert 0 <= a < 6, "invalid action"
        p = self.spec.p_intended
        slips = self._perpendicular_actions(a)
        slip_p = (1.0 - p) / 4.0
        # categorical draw: [intended] + [4 slips]
        choices = np.array([a] + slips, dtype=int)
        probs = np.array([p] + [slip_p] * 4, dtype=float)
        chosen = int(self.rng.choice(choices, p=probs))
        s_next = self._attempt_move(self._s, chosen)

        # rewards
        if s_next in self.terminals:
            r = self.terminals[s_next]
            done = True
        else:
            r = self.spec.step_cost
            done = False

        self._s = s_next
        return self.state2idx[self._s], r, done, {}

    # ----- Helpers -----
    def _attempt_move(self, s: Coord, a: int) -> Coord:
        v = self.ACTIONS[a]
        nxt = tuple(np.array(s) + v)
        return s if self._is_blocked(nxt) else nxt

    def _perpendicular_actions(self, a: int) -> List[int]:
        axis = a // 2  # 0=x, 1=y, 2=z
        other_axes = [ax for ax in (0, 1, 2) if ax != axis]
        res = []
        for ax in other_axes:
            res.extend([2 * ax, 2 * ax + 1])  # +axis and -axis
        return res  # exactly 4 actions

    def _is_blocked(self, s: Coord) -> bool:
        x, y, z = s
        return (
            x < 0 or x >= self.H or
            y < 0 or y >= self.W or
            z < 0 or z >= self.D or
            s in self.obstacles
        )

    def _build_state_index(self) -> None:
        self.state2idx: Dict[Coord, int] = {}
        self.idx2state: List[Coord] = []
        for z in range(self.D):
            for y in range(self.W):
                for x in range(self.H):
                    s = (x, y, z)
                    if s in self.obstacles:
                        continue
                    self.state2idx[s] = len(self.idx2state)
                    self.idx2state.append(s)

    def _generate_obstacles(self, ratio: float) -> Set[Coord]:
        """Generate ~ratio of cells as obstacles; reproducible via RNG."""
        total = self.H * self.W * self.D
        n_obs = int(round(total * ratio))
        all_cells = [(x, y, z) for z in range(self.D) for y in range(self.W) for x in range(self.H)]
        # sample without replacement
        obs_idx = self.rng.choice(len(all_cells), size=n_obs, replace=False)
        return {all_cells[i] for i in obs_idx}

    # ----- Convenience / debug -----
    def coord_of(self, idx: int) -> Coord:
        return self.idx2state[idx]

    def idx_of(self, coord: Coord) -> int:
        return self.state2idx[coord]
