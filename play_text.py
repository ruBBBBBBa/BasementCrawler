"""Play Crawler-v1 in the terminal.  w/a/s/d move, i/j/k/l shoot, space or . waits, q quits."""
import gymnasium as gym
import crawler  # noqa: F401

KEYS = {"w": 0, "s": 1, "a": 2, "d": 3, "i": 4, "k": 5, "j": 6, "l": 7, " ": 8, ".": 8}

env = gym.make("Crawler-v1", render_mode="ansi")
obs, info = env.reset()
total, done = 0.0, False
while not done:
    print(env.render(), f"\nobs {obs}  return {total:+.2f}")
    key = input("action> ")[:1] or " "
    if key == "q":
        break
    if key not in KEYS:
        continue
    obs, reward, terminated, truncated, info = env.step(KEYS[key])
    total += reward
    done = terminated or truncated
print(env.render(), f"\nfinished: return {total:+.2f}, exited {info['exited']}, hearts {info['hearts']}")
