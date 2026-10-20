"""Smoke test: can a minimal tabular Q-learner solve Crawler-v1?

Not the course implementation -- use the course's SARSA / Expected SARSA / Q-Learning
for the real experiments.
Usage: python smoke_train.py 300000          # train + evaluate 1, 2 and 3 enemies
       python smoke_train.py 300000 --watch  # train on 3 enemies, then watch it play
       python smoke_train.py 2000000 --v2    # train + evaluate Crawler-v2 (success = floor completed)
"""
import sys, time
import numpy as np
import gymnasium as gym
import crawler  # noqa: F401  (registers Crawler-v1)

def train(n_enemies, steps, alpha=0.1, gamma=0.99, eps_end=0.05, seed=0, env_id="Crawler-v1"):
    env = gym.make(env_id, **({"n_enemies": n_enemies} if env_id == "Crawler-v1" else {}))
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

def evaluate(Q, n_enemies, episodes=1000, env_id="Crawler-v1"):
    env = gym.make(env_id, **({"n_enemies": n_enemies} if env_id == "Crawler-v1" else {}))
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
        wins += info["exited"] or info.get("floor_completed", False)
        deaths += info["hearts"] <= 0
    return wins / episodes, deaths / episodes, np.mean(rets)

def watch(Q, n_enemies, episodes=5):
    """Show the greedy policy playing in a window."""
    env = gym.make("Crawler-v1", n_enemies=n_enemies, render_mode="human")
    for ep in range(episodes):
        obs, _ = env.reset(seed=20_000_000 + ep)
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(int(np.argmax(Q[obs])))
            done = term or trunc
        print(f"episode {ep + 1}: {'exited' if info['exited'] else 'failed'}")
        env.render()                                    # hold the last frame briefly
    env.close()


if __name__ == "__main__":
    numbers = [a for a in sys.argv[1:] if a.isdigit()]
    steps = int(numbers[0]) if numbers else 300_000
    if "--v2" in sys.argv:
        t0 = time.time()
        Q, _ = train(None, steps, env_id="Crawler-v2")
        w, d, r = evaluate(Q, None, env_id="Crawler-v2")
        print(f"Crawler-v2 steps={steps}: greedy floor completed {w:.1%}, death {d:.1%}, "
              f"mean return {r:.2f}, states touched {(np.abs(Q).sum(1) > 0).sum()}, {time.time() - t0:.0f}s")
        sys.exit(0)
    if "--watch" in sys.argv:                          # python smoke_train.py 300000 --watch
        Q, _ = train(3, steps)
        watch(Q, 3)
        sys.exit(0)
    for n in (1, 2, 3):
        t0 = time.time()
        Q, _ = train(n, steps)
        w, d, r = evaluate(Q, n)
        visited = (np.abs(Q).sum(1) > 0).sum()
        print(f"n_enemies={n} steps={steps}: greedy exit {w:.1%}, death {d:.1%}, "
              f"mean return {r:.2f}, states touched {visited}, {time.time() - t0:.0f}s")
