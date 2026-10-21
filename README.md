# Basement Crawler (working name)

A turn-based Gymnasium environment inspired by The Binding of Isaac, built for
Reinforcement Learning Assignment 1 (Option B) and trained with tabular SARSA,
Expected SARSA and Q-Learning.

The agent fights through rooms whose doors stay locked until every enemy is dead.
It shoots tears that travel across the room, searches the floor for the boss room,
and drops through the trapdoor to the next floor.

**Status:** v1 (one room) is built, tested and solved by all three algorithms. v2 (a fixed
five-room floor) is built and tested. With the −0.05 step reward, the first 1M-step runs
complete the floor in 84–94% of held-out greedy episodes. v3 is not started.

**Authors:** _name (email)_, _name (email)_

---

## Setup (first time)

You need Python 3.10–3.13 and Git. pygame has no build for Python 3.15 yet;
`mise.toml` pins 3.13.

```bash
git clone https://github.com/ruBBBBBBa/BasementCrawler.git
cd BasementCrawler

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt    # exact versions the team uses
pip install -e .                   # makes `import crawler` work from any folder

python check_v1.py                 # every check should pass
python -m pytest -q                # unit tests for the v1 and v2 environments
```

Activate the virtual environment (`source .venv/bin/activate`) every time you open
a new terminal for this project.

## Play

```bash
python play.py                     # play v2 (five rooms) in a window (keys below)
python play.py Crawler-v1          # play the one-room v1
python play_text.py                # v1 in the terminal: w/a/s/d move, i/j/k/l shoot
python check_v1.py                 # API check, mechanics tests, seeding, random baseline
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

## Train

`train.py` trains SARSA, Expected SARSA and Q-Learning. Every run starts from an all-zero
Q-table, so nothing carries over between runs.

```bash
python train.py --name E1                        # v1: 3 algorithms x 5 seeds, 300k steps each
python train.py --name V2 --env Crawler-v2       # v2: 3 algorithms x 5 seeds, 1M steps each
python train.py --name test --algo qlearning --seeds 1 --steps 50000
python train.py --name E2_hard --hearts 1 --enemy-move-prob 1.0
python train.py --name V2_old_step --env Crawler-v2 --step-reward -0.01
tensorboard --logdir runs                        # then open http://localhost:6006
```

Each run writes:
- `runs/<name>/<algo>/seed<n>/`: TensorBoard logs. The settings are under the Text tab.
- `qtables/<name>/<algo>_seed<n>.npy`: the final Q-table.
- `qtables/<name>/<algo>_seed<n>.json`: the settings and the final evaluation.

Reusing a name is refused; add `--overwrite` to replace it. Run `python train.py -h`
for every option.

| Setting | Default |
| --- | --- |
| `--algo` | `all` (or `sarsa`, `expected_sarsa`, `qlearning`) |
| `--seeds` | 5 |
| `--steps` | 300k (v1), 1M (v2) |
| Learning rate α / discount γ | 0.1 / 0.99 |
| Exploration ε | 1.0 → 0.05, linearly over the first 70% of steps |
| Evaluation | Greedy (ε = 0) on the same 100 held-out rooms, every 10k steps (v1) or 50k (v2) |

Update rules:
- **True terminal** (death, exit, floor completed): no bootstrap, so the target is just the reward.
- **Time-limit truncation:** bootstraps, because the timeout isn't part of the game.
- **Ties** between greedy actions are broken at random.

The TensorBoard curves that matter most:

| Tag | Shows |
| --- | --- |
| `eval/success_rate` | Greedy success on the held-out rooms (main comparison curve) |
| `eval/death_rate`, `eval/timeout_rate`, `eval/mean_return` | How the greedy agent fails, and its score |
| `eval/kills_per_episode` (v1), `eval/rooms_visited` (v2) | Progress per episode |
| `train/return`, `train/success_rate`, `train/timeout_rate` | Performance while still exploring |
| `train/epsilon` | The exploration schedule |
| `train/states_visited_pct` | Share of the state space seen so far |

## Watch and record a trained agent

```bash
python watch.py qtables/V2/sarsa_seed0.npy                            # play it in a window
python watch.py qtables/V2/sarsa_seed0.npy --record videos/v2_sarsa   # one .mp4 per episode
python watch.py --random --env Crawler-v2 --record videos/random      # random baseline
```

`watch.py` reads the `.json` next to the Q-table, so it opens the same version and settings
the agent was trained on. It plays rooms the agent never trained on.

## Use it from your own scripts

```python
import gymnasium as gym
import crawler                      # registers Crawler-v1 and Crawler-v2

env = gym.make("Crawler-v2")
obs, info = env.reset(seed=0)       # obs is an int in [0, env.observation_space.n)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

Size the Q-table with `env.observation_space.n` and `env.action_space.n` (9 actions).

Rendering:
- `render_mode="human"` opens a window that updates every step. Use it to watch a
  trained agent, never during training, because it slows every step to the frame rate.
- `render_mode="rgb_array"` returns each frame as an image, with no window, for
  `gymnasium.wrappers.RecordVideo`.
- `render_mode="ansi"` returns the room as text.

## Crawler-v1: one room

**Actions:** `0-3` move N/S/W/E, `4-7` shoot N/S/W/E, `8` wait.

**Observation:** `Discrete(5,650)`, one integer.
- While enemies are alive, it encodes the agent's cell, the nearest enemy's offset
  (row and column, each clipped to ±2), the tear cooldown, and how many enemies are left.
- Once the room is clear, it encodes only the agent's cell.

**Episode end:** the agent exits through an open door (success), loses all hearts,
or reaches 200 steps (truncated).

**`info` keys:** `hearts`, `enemies_left`, `cleared`, `exited`. Success is `exited`.

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

`gym.make("Crawler-v2")` (limit 500 steps). The rules are v1's, plus:
- **Doors.** Walking through an open door moves you to the next room, entering by the
  opposite door.
- **Enemies.** They spawn the first time you enter a room (at least 3 cells from you), and
  the doors lock until they're dead. A cleared room stays clear when you come back.
- **Hearts** carry over between rooms.
- **Rocks** (grey) block you, enemies and tears.
- **The boss** takes 3 hits. Killing it opens a trapdoor in the room centre; stepping onto
  it completes the floor and ends the episode (v2 has a single floor).
- **The minimap** (top-left corner) shows visited rooms; the boss door has a red frame.

**Observation:** `Discrete(29,675)`.
- Combat (enemies alive), 16,875 states: cell, nearest-enemy offset, cooldown, enemies
  left (boss HP in the boss room), hearts.
- Navigation (room clear), 12,800 states: cell, a label per door, trapdoor open.
  The door labels:
  - `WALL`: no door on that side.
  - `EXPLORED`: every room behind that door has been visited.
  - `UNEXPLORED_BRANCH`: some room behind that door, however deep, is unvisited.
  - `BOSS_NEXT_ROOM`: the door leads straight to the boss. This label takes priority
    over the others.
- Not observed: which room you are in, rocks, other enemies, the tear in flight.

**Rewards:** v1's, except:
- **Each step: −0.05** (the `step_reward` argument; v1 keeps −0.01).
- First entry into a room: +0.5 (the `enter_reward` argument).
- Boss killed: +10.
- Floor completed: +20.

**`info` keys:** v1's plus `room`, `rooms_visited`, `boss_hp`, `boss_defeated`,
`floor_completed`, `dead`. Success is `floor_completed`.

## Design notes

**Why v2 uses a −0.05 step reward.** With −0.01, timeouts that bootstrap and γ = 0.99,
hiding forever in a cleared room cost at most −1 in discounted terms, while a single hit
costs −3.
- **What went wrong:** Q-Learning learned to clear the easy dead-end room and then wander
  there until the time limit (0% success in every run). SARSA also collapsed on some seeds
  in long runs.
- **The fix:** at −0.05, stalling forever costs −5. SARSA improved, and Q-Learning started
  completing the floor.
- **Side effect:** a full 500-step timeout now totals −25, which is lower than dying
  (−19 from 3 lost hearts plus death). Deaths did not increase in our tests.

**Tried and dropped: transfer from v1.** We pretrained on v1 (300k steps), copied its
combat Q-values into v2's combat states, and continued on v2.
- **Result:** with the same amount of v2 training it did worse than training from
  scratch, and timeouts roughly doubled.
- **Likely cause:** the copied values kept their v1 meaning (a cleared room is followed by
  +5 and the end of the episode), so in v2, finishing a room looked like losing value.
- **Status:** the code was removed; the results are described in the report.

## Project layout

```
BasementCrawler/
├── crawler/
│   ├── __init__.py        # registers Crawler-v1 and Crawler-v2
│   ├── crawler_v1.py      # v1: one room (rules, observation, rewards)
│   ├── crawler_v2.py      # v2: fixed five-room floor, rocks, boss, trapdoor
│   └── render.py          # pygame graphics (window and video frames)
├── tests/
│   └── test_envs.py       # pytest unit tests for the v1 and v2 environments
├── outputs/smoke/         # screenshots of v2
├── play.py                # play in a window
├── play_text.py           # play v1 in the terminal
├── check_v1.py            # v1 validation checks
├── train.py               # SARSA / Expected SARSA / Q-Learning with TensorBoard
├── watch.py               # watch or record a saved Q-table
├── pyproject.toml         # package definition (pip install -e .)
├── requirements.txt       # pinned dependency versions
├── mise.toml              # pins Python 3.13
└── README.md
```

## Working together

```bash
git pull                              # always start from the latest main
git switch -c v3-generator            # one branch per piece of work
# ...edit, then run python check_v1.py and python -m pytest -q...
git add .
git commit -m "v3: random floor generator"
git push -u origin v3-generator       # then open a pull request on GitHub
```

- Merge into `main` only through pull requests, and only code that passes the checks.
- Adding a package? Run `pip install <package>`, then
  `pip freeze --exclude-editable > requirements.txt`, and commit `requirements.txt`.
- Keep TensorBoard logs (`runs/`), videos (`videos/`), Q-tables (`qtables/`, `*.npy`) and
  the virtual environment out of Git; `.gitignore` already covers them for `.venv/`.
  If your environment folder has another name, add it to `.gitignore`. Share results
  through Drive.

## Roadmap

| Version | Adds | States | Target |
| --- | --- | --- | --- |
| v1: one room | 1-3 chasers, travelling tears, doors that open when the room is clear | 5,650 | Done (20 Oct) |
| v2: fixed floor | Five rooms, rocks, boss with 3 hit points, trapdoor, hearts carried between rooms, door labels | 29,675 | Done (21 Oct) |
| v3: random floors | Isaac-style floor generator, dead ends, 2 floors, easy/medium/hard room pools | ~29,700 | 27 Oct |
| v4: enemies shoot | Enemy tears, combined move + shoot actions | TBD | If time allows |

Each version is registered as its own ID (`Crawler-v1`, `Crawler-v2`, ...), so earlier
versions keep working while later ones are built.

Assignment deadline: **2 Nov 2026, 23:59.**