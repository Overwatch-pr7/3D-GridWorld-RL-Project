# tests/test_env_day1_2.py
import math
import numpy as np
from src.gridworld import GridSpec, Gridworld3D

def approx(a, b, tol=0.03):
    return abs(a - b) <= tol

def run_many(env: Gridworld3D, start_coord, action, trials=20000):
    env._s = start_coord  # set internal state directly for this test
    hits = {}
    for _ in range(trials):
        # emulate one step but restore state after counting
        s_backup = env._s
        s2_idx, r, done, _ = env.step(action)
        s2 = env.coord_of(s2_idx)
        hits[s2] = hits.get(s2, 0) + 1
        env._s = s_backup  # restore
    return hits, trials

def main():
    spec = GridSpec(
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
    env = Gridworld3D(spec)

    # 1) Indexing & counts
    total_cells = spec.H * spec.W * spec.D
    n_obs_expected = round(total_cells * spec.obstacle_ratio)
    n_free = env.nS
    assert n_free == total_cells - n_obs_expected or abs(n_free - (total_cells - n_obs_expected)) <= 2, \
        f"Unexpected free count: got {n_free}"
    # Start, goal, pit must be free
    assert spec.start in env.state2idx
    assert spec.goal in env.state2idx
    assert spec.pit in env.state2idx

    # 2) Slip distribution test at a "safe" interior cell
    # Pick a coordinate that isn't near boundary to avoid blocked slips:
    # We'll search for one.
    interior = None
    for s in env.state2idx.keys():
        x,y,z = s
        if 1 <= x < spec.H-1 and 1 <= y < spec.W-1 and 1 <= z < spec.D-1 and s not in env.terminals:
            # also prefer no obstacles around (best effort)
            neighbors_clear = True
            for a in range(6):
                nxt = tuple(np.array(s) + env.ACTIONS[a])
                if nxt in env.obstacles:
                    neighbors_clear = False
                    break
            if neighbors_clear:
                interior = s
                break
    assert interior is not None, "Couldn't find a clean interior cell for slip test."
    # Choose intended +x (action=0). Expected probs: intended=0.8; each perpendicular=0.05
    hits, trials = run_many(env, interior, action=0, trials=20000)
    # Collect masses by type
    intended_target = tuple(np.array(interior) + env.ACTIONS[0])
    perps = [tuple(np.array(interior) + env.ACTIONS[a]) for a in env._perpendicular_actions(0)]

    p_intended_emp = hits.get(intended_target, 0) / trials
    p_perp_emp = [hits.get(p, 0) / trials for p in perps]
    # Some mass may be on "stay-put" if any perp was blocked; we selected a clear interior, so expect near-zero
    p_stay = hits.get(interior, 0) / trials

    assert approx(p_intended_emp, spec.p_intended, tol=0.02), f"Intended prob off: {p_intended_emp}"
    for i, pep in enumerate(p_perp_emp):
        assert approx(pep, (1-spec.p_intended)/4.0, tol=0.01), f"Perp prob {i} off: {pep}"
    assert p_stay < 0.01, f"Unexpected stay-put mass in interior: {p_stay}"

    # 3) Boundary test: from (0,0,0), take -x/-y/-z should stay with some prob mass
    env._s = (0,0,0)
    hits_b, trials_b = run_many(env, (0,0,0), action=1, trials=10000)  # -x intended
    # Many outcomes should be (0,0,0) due to blocked intended and some slips
    assert hits_b.get((0,0,0), 0) / trials_b > 0.15, "Expected noticeable stay-put at boundary"

    # 4) Terminal absorption & rewards
    # Move to goal location and take any action; should return +50 and done=True
    env._s = spec.goal
    s2_idx, r_goal, done_goal, _ = env.step(0)
    assert math.isclose(r_goal, 50.0), f"Goal reward mismatch: {r_goal}"
    assert done_goal, "Goal should terminate episode"
    # Pit
    env._s = spec.pit
    s2_idx, r_pit, done_pit, _ = env.step(0)
    assert math.isclose(r_pit, -50.0), f"Pit reward mismatch: {r_pit}"
    assert done_pit, "Pit should terminate episode"

    print("All Day 1–2 environment tests passed ✅")

if __name__ == "__main__":
    main()
