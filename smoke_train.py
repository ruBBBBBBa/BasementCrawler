"""Smoke test: can a minimal tabular Q-learner solve Crawler-v1?

Not the course implementation -- use the course's SARSA / Expected SARSA / Q-Learning
for the real experiments. Usage: python smoke_train.py 300000
"""
import sys, time
import numpy as np
import gymnasium as gym
import crawler  # noqa: F401  (registers Crawler-v1)

def train(n_enemies, steps, alpha=0.1, gamma=0.99, eps_end=0.05, seed=0):
    env = gym.make("Crawler-v1", n_enemies=n_enemies)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    ep = 0
    for t in range(steps):
        eps = max(eps_end, 1.0 - t / (0.7 * steps))
        a = int(rng.integers(9)) if rng.random() < eps else int(np.argmax(Q[obs]))
        nxt, r, term, trunc, _ = env.step(a)
        target = r if term else r + gamma * Q[nxt].max()
        Q[obs, a] += alpha * (target - Q[obs, a])
        obs = nxt
        if term or trunc:
            ep += 1
            obs, _ = env.reset(seed=seed * 100000 + ep)
    return Q, env

def evaluate(Q, n_enemies, episodes=1000):
    env = gym.make("Crawler-v1", n_enemies=n_enemies)
    wins = deaths = 0
    rets = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=10_000_000 + ep)
        done, total = False, 0.0
        while not done:
            obs, r, term, trunc, info = env.step(int(np.argmax(Q[obs])))
            total += r
            done = term or trunc
        rets.append(total)
        wins += info["exited"]
        deaths += info["hearts"] <= 0
    return wins / episodes, deaths / episodes, np.mean(rets)

if __name__ == "__main__":
    steps = int(sys.argv[1])
    for n in (1, 2, 3):
        t0 = time.time()
        Q, _ = train(n, steps)
        w, d, r = evaluate(Q, n)
        visited = (np.abs(Q).sum(1) > 0).sum()
        print(f"n_enemies={n} steps={steps}: greedy exit {w:.1%}, death {d:.1%}, "
              f"mean return {r:.2f}, states touched {visited}, {time.time() - t0:.0f}s")
