# Basement Crawler (working name)

A turn-based Gymnasium environment inspired by The Binding of Isaac, built for
Reinforcement Learning Assignment 1 (Option B) and trained with tabular SARSA,
Expected SARSA and Q-Learning.

The agent fights through rooms whose doors stay locked until every enemy is dead.
It shoots tears that travel across the room, searches the floor for the boss room,
and drops through the trapdoor to the next floor.

**Status:** v1 (one room) is built, tested and learnable. v2 (fixed five-room floor) is
built and tested; SARSA and Expected SARSA complete it in most held-out episodes at 1M
steps, Q-Learning does not (see `experiments/results/main_comparison.md`). v3 is not started.

**Authors:** _name (email)_, _name (email)_

---

## Setup (first time)

You need Python 3.10 or newer and Git.

```bash
git clone https://github.com/<your-user>/basement-crawler.git
cd basement-crawler

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt    # exact versions the team uses
pip install -e .                   # makes `import crawler` work from any folder
                                   # (also installs pygame for the graphics)

python check_v1.py                 # every check should pass
python -m pytest -q                # unit tests for v1 and v2
```

Activate the virtual environment (`source .venv/bin/activate`) every time you open
a new terminal for this project.

## Run

```bash
python play.py                     # play v2 (five rooms) in a window (keys below)
python play.py Crawler-v1          # play the one-room v1
python smoke_train.py 300000 --watch   # train a quick Q-learner, then watch it play
python check_v1.py                 # API check, mechanics tests, seeding, random baseline
python smoke_train.py 300000       # learnability check for 1, 2 and 3 enemies (~10 s each)
python smoke_train.py 2000000 --v2 # same check on Crawler-v2 (success = floor completed)
python play_text.py                # terminal version, handy over SSH or for debugging
```

Controls in `play.py` (the same layout as The Binding of Isaac). The game is turn-based:
each key press is one step.

| Key | Action |
| --- | --- |
| `W` / `A` / `S` / `D` | Move north / west / south / east |
| Arrow keys | Shoot in that direction |
| Space | Wait |
| `R` | Restart |
| `Esc` | Quit |

The top bar shows hearts, enemies left, and the tear cooldown (two dots, both lit when you
can shoot). Doors are barred until the room is clear, then turn into open doorways.

`smoke_train.py` is only a sanity check (it breaks greedy ties with `np.argmax`, which
always picks action 0 in untrained states). Use `train.py` for experiments.

## Train and evaluate

`train.py` runs SARSA, Expected SARSA or Q-Learning (adapted from the course's `deustorl`
code; see `crawler/tabular.py` for the differences: no bootstrap on true terminals,
bootstrap on time-limit truncation, seeded random tie-breaking).

```bash
python train.py --env Crawler-v1 --algo q_learning --steps 300000 --seed 0 --out runs/demo/v1
python evaluate.py runs/demo/v1                 # greedy, 1000 held-out seeds (1e9 ...)
python train.py --env Crawler-v2 --algo sarsa --steps 1000000 --seed 0 --out runs/demo/v2
python evaluate.py runs/demo/v2
tensorboard --logdir runs                        # curves: return, success, death, timeout,
                                                 # rooms visited, boss reached/defeated, ...
```

Each run directory holds `q.npy`, `config.json` (all settings, seeds, git commit),
`episodes.csv` (one row per training episode) and TensorBoard events; `evaluate.py` adds
`eval_<scenario>.json` / `.csv`. Defaults: alpha 0.1, gamma 0.99, epsilon 1.0 -> 0.05
linearly over the first 70% of steps.

Transfer and curriculum:
- `--init-from RUN` starts from another run's Q-table: same shape is copied; a
  Crawler-v1 table is mapped onto Crawler-v2's combat states (`crawler/transfer.py`),
  navigation states start at 0.
- `--scenario` picks a smaller Crawler-v2 floor ending at the boss (`crawler/scenarios.py`:
  `boss_only`, `hard_boss`, `medium_hard_boss`, `full`). Same observation/action encoding.
- `python experiments/transfer_curriculum.py --tag NAME --total 1000000 --v1-steps 300000`
  runs scratch vs transfer vs curriculum with equal total budgets, then evaluates and
  writes `experiments/results/NAME/` (run with `-h` for all options; it resumes).

## Use it from your own scripts

```python
import gymnasium as gym
import crawler                      # registers Crawler-v1

env = gym.make("Crawler-v1", n_enemies=3)
obs, info = env.reset(seed=0)       # obs is an int in [0, env.observation_space.n)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

Size the Q-table with `env.observation_space.n` (5,650 states in v1) and
`env.action_space.n` (9 actions). Log `info["exited"]` as the success rate.

Rendering:
- `render_mode="human"` opens a window that updates every step. Use it to watch a
  trained agent, never during training, because it slows every step to the frame rate.
- `render_mode="rgb_array"` returns each frame as an image, with no window, for
  `gymnasium.wrappers.RecordVideo`.
- `render_mode="ansi"` returns the room as text.

**Actions:** `0-3` move N/S/W/E, `4-7` shoot N/S/W/E, `8` wait.

**Observation:** one integer.
- While enemies are alive, it encodes the agent's cell, the nearest enemy's offset
  (row and column, each clipped to ±2), the tear cooldown, and how many enemies are left.
- Once the room is clear, it encodes only the agent's cell.

**Episode end:** the agent exits through an open door (success), loses all hearts,
or reaches 200 steps (truncated).

**`info` keys:** `hearts`, `enemies_left`, `cleared`, `exited`.

| Constructor argument | Default | Meaning |
| --- | --- | --- |
| `n_enemies` | 3 | Chasers in the room (1-3) |
| `enemy_move_prob` | 0.5 | Chance each enemy steps toward the agent each turn |
| `tear_speed` | 2 | Cells a tear moves per step |
| `tear_range` | 4 | Cells a tear flies before it lands |
| `fire_delay` | 2 | Steps between shots (changes the observation size) |
| `hearts` | 3 | Starting health |
| `render_mode` | None | `"human"` (window), `"rgb_array"` (images, for video) or `"ansi"` (text) |

| Event | Reward |
| --- | --- |
| Enemy killed | +1 |
| Room cleared | +2 |
| Exit through a door | +5 |
| Heart lost | −3 |
| Death | −10 |
| Each step | −0.01 |

## Crawler-v2: one fixed floor

```
. . . B .     S start (empty)        A medium: 2 chasers, 4 pillars
. . A C .     W easy dead end: 1     C hard: 3 chasers, centre rock
. W S . .     B boss: 3 HP, 2 rocks
```

`gym.make("Crawler-v2")` (limit 500 steps). Rules are v1's, plus:
- Walking through an open door moves you to the next room, entering by the opposite door.
  Enemies spawn the first time you enter a room (distance >= 3 from you); doors lock
  until they are dead. A room stays clear when you come back. Hearts carry over.
- Rocks (grey) block you, enemies and tears.
- The boss takes 3 hits. Killing it opens a trapdoor in the room centre; stepping onto it
  completes the floor and ends the episode (v2 has a single floor).
- The minimap (top-left corner) shows visited rooms; the boss door has a red frame.

**Observation:** `Discrete(29,675)`.
- Combat (enemies alive), 16,875 states: cell, nearest-enemy offset, cooldown, enemies
  left (boss HP in the boss room), hearts.
- Navigation (room clear), 12,800 states: cell, a label per door, trapdoor open.
  Labels: `WALL`, `EXPLORED`, `UNEXPLORED_BRANCH` (some room behind that door, however
  deep, is unvisited), `BOSS_NEXT_ROOM` (the door leads straight to the boss; this wins
  over the other labels).
- Not observed: which room you are in, rocks, other enemies, the tear in flight.

**Extra rewards:** first entry into a room +0.5 (`enter_reward` argument), boss killed
+10, floor completed +20.

**`info` keys:** v1's plus `room`, `rooms_visited`, `boss_hp`, `boss_defeated`,
`floor_completed`, `dead`. Success is `floor_completed`.

## Project layout

```
basement-crawler/
├── crawler/
│   ├── __init__.py        # registers Crawler-v1 and Crawler-v2
│   ├── crawler_v1.py      # the v1 environment (rules, observation, rewards)
│   ├── crawler_v2.py      # v2: fixed five-room floor, rocks, boss, trapdoor
│   ├── scenarios.py       # smaller v2 floors for curriculum learning
│   ├── tabular.py         # SARSA / Expected SARSA / Q-Learning updates
│   ├── transfer.py        # v1 -> v2 Q-table mapping
│   └── render.py          # pygame graphics (window and video frames)
├── play.py                # play in a window
├── play_text.py           # play in the terminal
├── check_v1.py            # validation checks
├── tests/                 # pytest unit tests (envs + RL code)
├── train.py               # train one stage, save Q-table/config/TensorBoard
├── evaluate.py            # greedy evaluation on held-out seeds
├── experiments/           # experiment runners; results/ holds summaries
├── smoke_train.py         # quick learnability check, --watch to see it play
├── pyproject.toml         # package definition (pip install -e .)
├── requirements.txt       # pinned dependency versions
└── README.md
```

## Working together

```bash
git pull                              # always start from the latest main
git switch -c v2-floor                # one branch per piece of work
# ...edit, then run python check_v1.py...
git add .
git commit -m "v2: multi-room floor"
git push -u origin v2-floor           # then open a pull request on GitHub
```

- Merge into `main` only through pull requests, and only code that passes the checks.
- Adding a package? Run `pip install <package>`, then
  `pip freeze --exclude-editable > requirements.txt`, and commit `requirements.txt`.
- Keep Tensorboard logs (`runs/`), videos (`videos/`), saved Q-tables (`*.npy`) and
  `.venv/` out of Git; `.gitignore` already does this. Share them through Drive.

## Roadmap

| Version | Adds | States | Target |
| --- | --- | --- | --- |
| v1: one room | 1-3 chasers, travelling tears, doors that open when the room is clear | 5,650 | Done (20 Oct) |
| v2: fixed floor | Several rooms, rocks, boss with 3 hit points, trapdoor, hearts carried between rooms, door labels | ~29,700 | 24 Oct |
| v3: random floors | Isaac-style floor generator, dead ends, 2 floors, easy/medium/hard room pools | ~29,700 | 27 Oct |
| v4: enemies shoot | Enemy tears, combined move + shoot actions | TBD | If time allows |

Each version is registered as its own ID (`Crawler-v1`, `Crawler-v2`, ...), so earlier
versions keep working while later ones are built.

Assignment deadline: **2 Nov 2026, 23:59.**
