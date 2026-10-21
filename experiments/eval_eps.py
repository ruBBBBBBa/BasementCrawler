"""Diagnostic: re-evaluate every final Q-table of an experiment tag with an epsilon-greedy
policy, next to the primary greedy result.  python experiments/eval_eps.py main_T3000k 0.05 500"""
import csv
import json
import os
import sys
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from evaluate import evaluate_run  # noqa: E402

tag, eps, episodes = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
res = os.path.join(ROOT, "experiments", "results", tag)
rows = list(csv.DictReader(open(os.path.join(res, "runs.csv"))))


def job(row):
    out = os.path.join(ROOT, row["run"], f"eval_full_eps{eps}.json")
    s = json.load(open(out)) if os.path.exists(out) else evaluate_run(os.path.join(ROOT, row["run"]), "full", episodes, eps=eps)
    return {**{k: row[k] for k in ("condition", "algo", "seed", "success_rate", "timeout_rate")},
            f"eps{eps}_success_rate": s["success_rate"], f"eps{eps}_timeout_rate": s["timeout_rate"],
            f"eps{eps}_dead_rate": s["dead_rate"], f"eps{eps}_boss_reached_rate": s["boss_reached_rate"],
            f"eps{eps}_episodes": s["episodes"]}


if __name__ == "__main__":
    with Pool(10) as pool:
        out = pool.map(job, rows, chunksize=1)
    with open(os.path.join(res, f"eval_eps{eps}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {len(out)} rows")
