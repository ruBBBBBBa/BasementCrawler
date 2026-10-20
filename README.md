# Basement Crawler (working name)

Turn-based, Binding of Isaac-inspired Gymnasium environment for RL Assignment 1, Option B.
This folder holds **v1**: one 5x5 room, 1-3 chasing enemies, travelling tears, doors that
open once the room is clear.

## Install

```bash
pip install gymnasium numpy          # Python 3.10+
```

Run every command from this folder so `import crawler` finds the package.

## Run

```bash
python check_v1.py                   # API check, mechanics tests, seeding, random baseline
python play_text.py                  # play it yourself: w/a/s/d move, i/j/k/l shoot, space waits
python smoke_train.py 300000         # quick Q-learning smoke test (about 10 s per setting)
```

## Use from your own scripts

```python
import gymnasium as gym
import crawler                       # registers Crawler-v1

env = gym.make("Crawler-v1", n_enemies=3, enemy_move_prob=0.5)
obs, info = env.reset(seed=0)        # obs is an int in [0, env.observation_space.n)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

`info` carries `hearts`, `enemies_left`, `cleared` and `exited` (the success flag to log).

| Constructor argument | Default | Meaning |
| --- | --- | --- |
| `n_enemies` | 3 | Chasers in the room (1-3) |
| `enemy_move_prob` | 0.5 | Chance each enemy steps toward the agent each turn |
| `tear_speed` | 2 | Cells a tear moves per step |
| `tear_range` | 4 | Cells a tear flies before it lands |
| `fire_delay` | 2 | Steps between shots (changes the observation size) |
| `hearts` | 3 | Starting health |
| `render_mode` | None | `"ansi"` (text) or `"rgb_array"` (image, for video) |

Actions: `0-3` move N/S/W/E, `4-7` shoot N/S/W/E, `8` wait. Episodes are truncated at 200 steps.
