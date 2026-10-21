"""Watch a trained agent play in a window, or record it to video for the report.

    python watch.py qtables/E1/qlearning_seed0.npy                          # window
    python watch.py qtables/E1/qlearning_seed0.npy --record videos/qlearning
    python watch.py --random --record videos/random                         # baseline
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import gymnasium as gym

import crawler  # noqa: F401  (registers Crawler-v1)

p = argparse.ArgumentParser()
p.add_argument("qtable", nargs="?", help="a .npy file saved by train.py")
p.add_argument("--random", action="store_true", help="random agent instead of a Q-table")
p.add_argument("--episodes", type=int, default=5)
p.add_argument("--record", metavar="FOLDER", help="save one .mp4 per episode instead of a window")
args = p.parse_args()
if not args.qtable and not args.random:
    p.error("give a Q-table file or --random")

env_kwargs, Q = {}, None
if args.qtable:
    Q = np.load(args.qtable)
    config = Path(args.qtable).with_suffix(".json")
    if config.exists():                          # same room settings the agent was trained on
        env_kwargs = json.loads(config.read_text())["env"]

env = gym.make("Crawler-v1", render_mode="rgb_array" if args.record else "human", **env_kwargs)
if args.record:
    env = gym.wrappers.RecordVideo(env, args.record, episode_trigger=lambda e: True,
                                   name_prefix=Path(args.qtable).stem if args.qtable else "random")
rng = np.random.default_rng(0)
for ep in range(args.episodes):
    s, _ = env.reset(seed=30_000_000 + ep)       # rooms the agent never trained on
    done, total = False, 0.0
    while not done:
        if Q is None:
            a = int(rng.integers(9))
        else:
            a = int(rng.choice(np.flatnonzero(Q[s] == Q[s].max())))
        s, r, term, trunc, info = env.step(a)
        total, done = total + r, term or trunc
    print(f"episode {ep + 1}: {'exited' if info['exited'] else 'died' if info['hearts'] <= 0 else 'time up'}, "
          f"return {total:+.2f}")
    if not args.record:
        time.sleep(1.0)                          # keep the last frame on screen briefly
env.close()
if args.record:
    print(f"videos saved in {args.record}/")