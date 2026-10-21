"""Train SARSA, Expected SARSA and Q-Learning on Basement Crawler, logging to TensorBoard.

Every run starts from an all-zero Q-table: nothing is carried over between runs.

Examples:
    python train.py --name E1   # all 3 algorithms, 5 seeds each
    python train.py --name E2_hard --hearts 1 --enemy-move-prob 1.0
    python train.py --name test --algo qlearning --seeds 1 --steps 50000
    tensorboard --logdir runs   # then open http://localhost:6006
"""
import argparse
import json
import shutil
import time
from collections import deque
from pathlib import Path

import numpy as np
import gymnasium as gym

import crawler

try:
    from tensorboardX import SummaryWriter
except ImportError:                              # same API if PyTorch is installed instead
    from torch.utils.tensorboard import SummaryWriter

ALGOS = ("sarsa", "expected_sarsa", "qlearning")
EVAL_SEED = 10_000_000                           # every run is evaluated on the same rooms


def epsilon_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(Q.shape[1]))
    best = np.flatnonzero(Q[s] == Q[s].max())
    return int(best[0] if best.size == 1 else best[rng.integers(best.size)])  # random tie-break


def bootstrap(Q, s_next, a_next, algo, eps):
    #Value of the next state that each algorithm uses in its update
    q = Q[s_next]
    if algo == "sarsa":    # on-policy: the action actually taken next
        return q[a_next]
    if algo == "qlearning":    # off-policy: the best action
        return q.max()
    greedy = np.flatnonzero(q == q.max())   # expected SARSA: average under eps-greedy
    probs = np.full(q.size, eps / q.size)
    probs[greedy] += (1 - eps) / greedy.size
    return float(probs @ q)


def evaluate(Q, env_kwargs, episodes):
    #Greedy policy (epsilon = 0) on a fixed set of rooms.
    env = gym.make("Crawler-v1", **env_kwargs)
    rng = np.random.default_rng(0)
    wins = deaths = 0
    returns, lengths = [], []
    for ep in range(episodes):
        s, _ = env.reset(seed=EVAL_SEED + ep)
        done, total, steps = False, 0.0, 0
        while not done:
            s, r, term, trunc, info = env.step(epsilon_greedy(Q, s, 0.0, rng))
            total, steps, done = total + r, steps + 1, term or trunc
        wins += info["exited"]
        deaths += info["hearts"] <= 0
        returns.append(total)
        lengths.append(steps)
    return {"success_rate": wins / episodes, "death_rate": deaths / episodes,
            "mean_return": float(np.mean(returns)), "mean_length": float(np.mean(lengths))}


def train(algo, seed, args, env_kwargs, run_dir):
    env = gym.make("Crawler-v1", **env_kwargs)
    n_states, n_actions = env.observation_space.n, env.action_space.n
    Q = np.zeros((n_states, n_actions))
    # The Q-table, initialized to zeros: one row per observation and one column per action (9).
    visited = np.zeros(n_states, dtype=bool)
    rng = np.random.default_rng(seed)
    writer = SummaryWriter(str(run_dir))
    config = {"algo": algo, "seed": seed, "steps": args.steps, "alpha": args.alpha,
              "gamma": args.gamma, "eps_start": args.eps_start, "eps_end": args.eps_end,
              "eps_decay": args.eps_decay, "env": env_kwargs}
    writer.add_text("config", "```\n" + json.dumps(config, indent=2) + "\n```", 0)

    decay_steps = max(1, int(args.eps_decay * args.steps))
    # The trainer begins with a high exploration rate (eps_start = 1.0) and gradually reduces it
    # to a minimum (eps_end = 0.05) over the first 70% of training (--eps-decay).

    def eps_at(t):   # linear decay, then constant
        return max(args.eps_end, args.eps_start - (args.eps_start - args.eps_end) * t / decay_steps)

    recent = deque(maxlen=100)    # successes of the last 100 episodes
    window = []    # episodes finished since the last log point
    ep, ep_return, ep_len = 0, 0.0, 0
    s, _ = env.reset(seed=seed * 1_000_000)
    eps = eps_at(0)
    a = epsilon_greedy(Q, s, eps, rng)
    for t in range(1, args.steps + 1):
        s2, r, term, trunc, info = env.step(a)
        visited[s] = True
        eps = eps_at(t)
        a2 = epsilon_greedy(Q, s2, eps, rng)
        # terminal: no future value. Truncated (time limit): the game isn't over, so bootstrap.
        target = r if term else r + args.gamma * bootstrap(Q, s2, a2, algo, eps)
        Q[s, a] += args.alpha * (target - Q[s, a])
        # Q[s, a] estimates the expected long-term discounted return of taking action a in observation s 
        # and following the policy afterwards (the eps-greedy one for SARSA and Expected SARSA, the greedy one for Q-Learning).
        s, a = s2, a2
        ep_return, ep_len = ep_return + r, ep_len + 1

        if term or trunc:
            ep += 1
            recent.append(info["exited"])
            window.append((ep_return, ep_len, info["exited"], info["hearts"] <= 0,
                           env_kwargs["n_enemies"] - info["enemies_left"]))
            s, _ = env.reset(seed=seed * 1_000_000 + ep)
            a = epsilon_greedy(Q, s, eps, rng)
            ep_return, ep_len = 0.0, 0

        if t % args.log_every == 0 and window:   # averages over the episodes in this window
            returns, lengths, wins, died, kills = np.array(window, dtype=float).T
            writer.add_scalar("train/return", returns.mean(), t)
            writer.add_scalar("train/episode_length", lengths.mean(), t)
            writer.add_scalar("train/success_rate", wins.mean(), t)
            writer.add_scalar("train/success_rate_last100", np.mean(recent), t)
            writer.add_scalar("train/death_rate", died.mean(), t)
            writer.add_scalar("train/kills_per_episode", kills.mean(), t)
            writer.add_scalar("train/epsilon", eps, t)
            window.clear()

        if t % args.eval_every == 0 or t == args.steps:
            res = evaluate(Q, env_kwargs, args.eval_episodes)
            for key, value in res.items():
                writer.add_scalar(f"eval/{key}", value, t)
            writer.add_scalar("train/states_visited_pct", 100 * visited.mean(), t)
            writer.add_scalar("train/episodes", ep, t)

    writer.close()
    return Q, res, config


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="experiment name, e.g. E1")
    p.add_argument("--algo", choices=ALGOS + ("all",), default="all")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay", type=float, default=0.7, help="fraction of steps spent decaying")
    p.add_argument("--log-every", type=int, default=1_000, help="steps between training log points")
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--n-enemies", type=int, default=3)
    p.add_argument("--enemy-move-prob", type=float, default=0.5)
    p.add_argument("--hearts", type=int, default=3)
    p.add_argument("--tear-range", type=int, default=4)
    p.add_argument("--tear-speed", type=int, default=2)
    p.add_argument("--overwrite", action="store_true", help="delete earlier results with this name")
    args = p.parse_args()

    env_kwargs = {"n_enemies": args.n_enemies, "enemy_move_prob": args.enemy_move_prob,
                  "hearts": args.hearts, "tear_range": args.tear_range, "tear_speed": args.tear_speed}
    log_root, q_root = Path("runs") / args.name, Path("qtables") / args.name
    if log_root.exists() or q_root.exists():
        if not args.overwrite:
            raise SystemExit(f"'{args.name}' already exists. Pick another --name or add --overwrite.")
        shutil.rmtree(log_root, ignore_errors=True)
        shutil.rmtree(q_root, ignore_errors=True)
    q_root.mkdir(parents=True)

    algos = ALGOS if args.algo == "all" else (args.algo,)
    summary = {}
    for algo in algos:
        for seed in range(args.seeds):
            t0 = time.time()
            Q, res, config = train(algo, seed, args, env_kwargs, log_root / algo / f"seed{seed}")
            np.save(q_root / f"{algo}_seed{seed}.npy", Q)
            (q_root / f"{algo}_seed{seed}.json").write_text(json.dumps({**config, "final_eval": res}, indent=2))
            summary.setdefault(algo, []).append(res["success_rate"])
            print(f"{algo:15s} seed {seed}: success {res['success_rate']:6.1%}, "
                  f"death {res['death_rate']:6.1%}, return {res['mean_return']:6.2f}  ({time.time() - t0:.0f}s)")

    print("\nFinal greedy success rate (mean +- std over seeds):")
    for algo, rates in summary.items():
        print(f"  {algo:15s} {np.mean(rates):6.1%} +- {np.std(rates):.1%}")
    print(f"\nLogs: runs/{args.name}   Q-tables: qtables/{args.name}   View: tensorboard --logdir runs")


if __name__ == "__main__":
    main()