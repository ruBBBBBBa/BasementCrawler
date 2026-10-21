"""Does v1 pretraining or a v2 curriculum beat Crawler-v2 trained from scratch?

Conditions, for a total budget T and a v1 pretraining budget P (environment steps):
    scratch        Crawler-v2 full floor, T steps
    scratch_short  Crawler-v2 full floor, T-P steps (same v2 steps as transfer_v1)
    transfer_v1    Crawler-v1 (3 chasers) P steps -> map combat rows to v2 -> v2 full, T-P steps
    curriculum     v2 boss_only 10% -> hard_boss 20% -> medium_hard_boss 20% -> full 50% of T
Every stage uses the same alpha/gamma and restarts the same epsilon schedule. The final
Q-table of each run is evaluated greedily on the full v2 floor, on the held-out seeds of
evaluate.py. Finished stages/evaluations are skipped, so an interrupted run resumes.

    python experiments/transfer_curriculum.py --tag pilot --total 1000000 --v1-steps 300000 \
        --seeds 3 --algos q_learning --episodes 300
Writes runs/<tag>/... (checkpoints, TensorBoard) and experiments/results/<tag>/ (summaries).
"""
import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from evaluate import evaluate_run  # noqa: E402
from train import code_version, train  # noqa: E402

CURRICULUM = [("boss_only", 0.1), ("hard_boss", 0.2), ("medium_hard_boss", 0.2), ("full", 0.5)]
METRICS = ["success_rate", "boss_reached_rate", "boss_defeated_rate", "dead_rate",
           "timeout_rate", "return_mean", "length_mean", "rooms_visited_mean"]


def conditions(total, v1_steps):
    return {
        "scratch": [("Crawler-v2", "full", total)],
        "scratch_short": [("Crawler-v2", "full", total - v1_steps)],
        "transfer_v1": [("Crawler-v1", None, v1_steps), ("Crawler-v2", "full", total - v1_steps)],
        "curriculum": [("Crawler-v2", name, int(frac * total)) for name, frac in CURRICULUM],
    }


def run_job(job):
    tag, cond, stages, algo, seed, episodes, hp = job
    base = os.path.join(ROOT, "runs", tag, cond, algo, f"seed{seed}")
    parent = None
    for i, (env_id, scenario, steps) in enumerate(stages):
        out = os.path.join(base, f"{i}_{env_id[-2:]}_{scenario or 'full'}")
        if not os.path.exists(os.path.join(out, "config.json")):
            train(env_id, scenario, {}, algo, steps, seed=100 * seed + i, init_from=parent,
                  out=out, **hp)
        parent = out
    eval_file = os.path.join(parent, "eval_full.json")
    if os.path.exists(eval_file):
        summary = json.load(open(eval_file))
    else:
        summary = evaluate_run(parent, "full", episodes)
    cfg = json.load(open(os.path.join(parent, "config.json")))
    v2_steps = sum(s for e, _, s in stages if e == "Crawler-v2")
    print(f"{cond:14s} {algo:15s} seed {seed}: success {summary['success_rate']:.1%} "
          f"boss reached {summary['boss_reached_rate']:.1%}", flush=True)
    return {"condition": cond, "algo": algo, "seed": seed, "total_steps": cfg["total_steps"],
            "v2_steps": v2_steps, "run": os.path.relpath(parent, ROOT),
            **{m: summary[m] for m in METRICS}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True)
    p.add_argument("--total", type=int, default=1_000_000)
    p.add_argument("--v1-steps", type=int, default=300_000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--algos", default="q_learning", help="comma list")
    p.add_argument("--conditions", default="scratch,scratch_short,transfer_v1,curriculum")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    a = p.parse_args()
    hp = {"alpha": a.alpha, "gamma": a.gamma}
    conds = conditions(a.total, a.v1_steps)
    jobs = [(a.tag, c, conds[c], algo, s, a.episodes, hp) for c in a.conditions.split(",")
            for algo in a.algos.split(",") for s in range(a.seeds)]
    with Pool(a.workers) as pool:
        rows = pool.map(run_job, jobs, chunksize=1)

    res = os.path.join(ROOT, "experiments", "results", a.tag)
    os.makedirs(res, exist_ok=True)
    with open(os.path.join(res, "protocol.json"), "w") as f:
        json.dump({**vars(a), "stages": {c: conds[c] for c in a.conditions.split(",")},
                   "eval": f"greedy, full v2 floor, seeds 1e9 .. 1e9+{a.episodes - 1}",
                   "code": code_version()}, f, indent=2)
    with open(os.path.join(res, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [f"# {a.tag}: mean ± std over {a.seeds} seeds, {a.episodes} held-out greedy episodes each",
             "", "| condition | algo | total steps | v2 steps | " + " | ".join(METRICS) + " |",
             "|" + "---|" * (4 + len(METRICS))]
    for c in a.conditions.split(","):
        for algo in a.algos.split(","):
            sel = [r for r in rows if r["condition"] == c and r["algo"] == algo]
            cells = [f"{np.mean([r[m] for r in sel]):.3f} ± {np.std([r[m] for r in sel]):.3f}"
                     for m in METRICS]
            lines.append(f"| {c} | {algo} | {sel[0]['total_steps']:,} | {sel[0]['v2_steps']:,} | "
                         + " | ".join(cells) + " |")
    with open(os.path.join(res, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
