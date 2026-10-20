"""Validation checks for Crawler-v1. Run from the project folder: python check_v1.py"""
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import crawler  # noqa: F401  (registers Crawler-v1)
from crawler.crawler_v1 import CrawlerV1Env

# 1. Gymnasium API check -------------------------------------------------------
check_env(gym.make("Crawler-v1", render_mode="rgb_array").unwrapped)
print("1. check_env passed")

# 2. Scripted mechanics tests ---------------------------------------------------
env = CrawlerV1Env(n_enemies=1)
env.reset(seed=0)

env.agent, env.enemies = (2, 2), [(0, 2)]          # enemy two cells north
env.enemy_move_prob = 0.0                          # freeze enemies for the test
_, r, *_ = env.step(4)                             # shoot north
assert env.enemies == [] and env.cleared, "tear should kill an enemy 2 cells away in one step"
print(f"2a. two-cell shot kills in one step (reward {r:+.2f})")

env.reset(seed=0)
env.agent, env.enemies, env.enemy_move_prob = (4, 0), [(0, 0)], 0.0
env.step(4)                                        # tear covers rows 3, 2
assert env.enemies == [(0, 0)] and env.tear is not None
env.step(8)                                        # tear covers rows 1, 0
assert env.enemies == [], "enemy at range 4 should die on the second step"
print("2b. range-4 shot lands on the second step")

env.reset(seed=0)
env.agent, env.enemies, env.enemy_move_prob = (2, 2), [(0, 0)], 0.0
env.step(4)                                        # fire
env.step(5)                                        # immediate second shot is ignored
assert env.tear is None or env.tear[1] == 0, "cooldown should block the second shot"
print("2c. cooldown blocks a second shot")

env.reset(seed=0)
env.agent, env.enemies, env.enemy_move_prob = (1, 2), [(0, 2)], 0.0
hearts = env.hearts
env.step(0)                                        # walk into the enemy
assert env.hearts == hearts - 1 and env.agent == (1, 2)
print("2d. walking into an enemy costs a heart and blocks the move")

env.reset(seed=0)
env.agent, env.enemies, env.enemy_move_prob = (4, 2), [(0, 0)], 0.0
_, _, term, *_ = env.step(1)                       # south door while locked
assert not term and env.agent == (4, 2)
env.enemies, env.cleared = [], True
_, r, term, *_ = env.step(1)                       # door open now
assert term and env.exited
print(f"2e. door locked until clear, then exit ends the episode (reward {r:+.2f})")

# 3. Seeding: same seed + same actions -> identical episode --------------------
def rollout(seed):
    e = gym.make("Crawler-v1")
    obs, _ = e.reset(seed=seed)
    rng = np.random.default_rng(123)
    trace = [obs]
    for _ in range(200):
        obs, r, term, trunc, _ = e.step(int(rng.integers(9)))
        trace += [obs, r]
        if term or trunc:
            break
    return trace

assert rollout(7) == rollout(7)
print("3. same seed reproduces the same episode")

# 4. Random agent baseline and observation bounds ------------------------------
e = gym.make("Crawler-v1")
returns, lengths, wins, deaths, seen = [], [], 0, 0, set()
for ep in range(2000):
    obs, _ = e.reset(seed=ep)
    total, steps, done = 0.0, 0, False
    while not done:
        assert e.observation_space.contains(obs)
        seen.add(obs)
        obs, r, term, trunc, info = e.step(e.action_space.sample())
        total, steps, done = total + r, steps + 1, term or trunc
    returns.append(total)
    lengths.append(steps)
    wins += info["exited"]
    deaths += info["hearts"] <= 0
n = e.observation_space.n
print(f"4. random agent over 2000 episodes: mean return {np.mean(returns):.2f}, "
      f"exits {wins / 20:.1f}%, deaths {deaths / 20:.1f}%, mean length {np.mean(lengths):.0f} steps")
print(f"   observation space {n} states, {len(seen)} visited ({100 * len(seen) / n:.0f}%)")
