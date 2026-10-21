"""Greedy (epsilon = 0) evaluation of a saved Q-table on held-out seeds.

    python evaluate.py runs/demo/v1                      # 1000 episodes, seeds 1e9 .. 1e9+999
    python evaluate.py runs/demo/t --scenario full --episodes 300

Evaluation reset seeds start at EVAL_SEED_BASE = 1e9; training resets never reach it.
Greedy ties are broken at random by a generator seeded with the episode's reset seed, so
an evaluation is exactly repeatable (--tie-break first uses np.argmax instead, as the old
smoke_train.py did; kept only as a diagnostic). --eps > 0 evaluates an epsilon-greedy
policy instead (diagnostic for greedy loops; the reported metric stays eps = 0). The Q-table is opened read-only and never changed.
Writes eval_<scenario>.json (summary) and eval_<scenario>.csv (per episode) to the run.
"""
import argparse
import csv
import json
import os

import numpy as np

from crawler.tabular import epsilon_greedy, greedy
from train import EPISODE_FIELDS, episode_outcome, in_boss_room, load_run, make_env

EVAL_SEED_BASE = 10**9
EVAL_FIELDS = ["seed"] + [f for f in EPISODE_FIELDS if f not in ("episode", "step", "epsilon",
                                                                  "rows_visited_frac")]


def evaluate(q, env_id, scenario=None, env_kwargs=None, episodes=1000, seed_base=EVAL_SEED_BASE,
             tie_break="random", eps=0.0):
    q = np.asarray(q).view()                     # read-only view; caller's array untouched
    q.setflags(write=False)
    env = make_env(env_id, scenario, env_kwargs)
    assert q.shape == (env.observation_space.n, env.action_space.n), "checkpoint/env mismatch"
    pick = greedy if tie_break == "random" else lambda row, rng: int(np.argmax(row))
    if eps > 0:
        pick = lambda row, rng: epsilon_greedy(row, eps, rng)
    rows = []
    for i in range(episodes):
        seed = seed_base + i
        rng = np.random.default_rng(seed)
        obs, info = env.reset(seed=seed)
        ret, length, boss_seen, done = 0.0, 0, in_boss_room(env), False
        while not done:
            obs, r, terminated, truncated, info = env.step(pick(q[obs], rng))
            ret, length = ret + r, length + 1
            boss_seen = boss_seen or in_boss_room(env)
            done = terminated or truncated
        rows.append({"seed": seed, "return": round(ret, 4), "length": length,
                     "boss_reached": boss_seen, **episode_outcome(info, terminated, truncated)})
    env.close()
    return summarize(rows), rows


def summarize(rows):
    col = lambda k: np.array([r[k] for r in rows], dtype=float)
    visits = col("rooms_visited").astype(int)
    return {"episodes": len(rows),
            **{f"{k}_rate": float(col(k).mean()) for k in
               ("success", "dead", "timeout", "boss_reached", "boss_defeated")},
            "return_mean": float(col("return").mean()), "return_std": float(col("return").std()),
            "length_mean": float(col("length").mean()),
            "rooms_visited_mean": float(visits.mean()),
            "rooms_visited_hist": {int(k): int(v) for k, v in zip(*np.unique(visits, return_counts=True))}}


def evaluate_run(run_dir, scenario="config", episodes=1000, seed_base=EVAL_SEED_BASE,
                 tie_break="random", eps=0.0):
    q, cfg = load_run(run_dir)
    scenario = cfg["scenario"] if scenario == "config" else scenario
    summary, rows = evaluate(q, cfg["env_id"], scenario, cfg["env_kwargs"], episodes, seed_base,
                             tie_break, eps)
    summary.update(run=run_dir, env_id=cfg["env_id"], scenario=scenario or "full", tie_break=tie_break,
                   eps=eps,
                   seeds=f"{seed_base}..{seed_base + episodes - 1}",
                   success_means="floor_completed" if cfg["env_id"] == "Crawler-v2" else "exited")
    name = (f"eval_{scenario or 'full'}" + ("" if tie_break == "random" else f"_{tie_break}")
            + (f"_eps{eps}" if eps > 0 else ""))
    with open(os.path.join(run_dir, name + ".json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(run_dir, name + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, EVAL_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--scenario", default="config", help="default: the scenario the run trained on")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--tie-break", default="random", choices=["random", "first"])
    p.add_argument("--eps", type=float, default=0.0)
    a = p.parse_args()
    s = evaluate_run(a.run, a.scenario, a.episodes, tie_break=a.tie_break, eps=a.eps)
    print(json.dumps(s, indent=2))
