# Basement Crawler

A turn-based, grid-based dungeon crawler built as a custom **Gymnasium** environment for **Reinforcement Learning — Assignment 1 (Option B: Environment Design)**. The game is inspired by the room progression and combat mechanics of *The Binding of Isaac*, but uses its own discrete rules and Pygame interface.

We developed three versions of the environment and compared three tabular reinforcement learning algorithms: **SARSA**, **Expected SARSA**, and **Q-Learning**.

**Authors:** Rubén García (ruben.garcia.saiz@opendeusto.es) and Aner Junquera (aner.junquera@opendeusto.es).

**Project status:** Completed. `Crawler-v3` is the final environment.

## 1. Environment overview

The player moves through rooms, defeats enemies by shooting tears, opens doors by clearing each room, and searches for the boss. Defeating a boss opens a trapdoor. In the final version, the agent must complete **two procedurally generated floors** to win.

| Version | Main features | Observation states | Episode limit |
| --- | --- | ---: | ---: |
| `Crawler-v1` | One 5×5 room with chasing enemies and an exit | 5,650 | 200 steps |
| `Crawler-v2` | Fixed five-room floor, rocks, boss and trapdoor | 29,675 | 500 steps |
| `Crawler-v3` | Two procedurally generated floors, room pools and persistent health | 20,925 | 1,000 steps |

All three environments use `Discrete(9)` actions: four movements, four shooting directions and one wait action.

### Main mechanics

- **Movement:** move one cell north, south, west or east on a 5×5 room grid.
- **Combat:** shoot tears in four directions. Tears travel up to four cells, move two cells per step, and have a two-step firing cooldown.
- **Enemies:** chasing enemies attempt to move toward the player with a default probability of 0.5 per turn. Enemy contact costs one heart; the player starts with three hearts.
- **Doors:** locked while enemies remain in the current room. Cleared rooms remain cleared when revisited.
- **Boss:** has three hit points and opens a trapdoor when defeated.
- **Floors:** v3 generates new room layouts each episode (four rooms on floor 1 and five on floor 2 by default). Health carries over between floors, and each defeated boss restores one heart, up to the starting maximum.

### Observations

The agent receives **one integer observation**, not the rendered image. Its information is compacted into two modes:

- **Combat:** player position, relative position of the nearest enemy, firing cooldown, remaining enemies or boss health, and—in v2/v3—player health.
- **Navigation:** player position, available doors and whether unexplored branches or a boss lie behind them, plus trapdoor availability.

The v3 navigation representation merges walls and fully explored branches into a single door label, reducing its total observation space to 20,925 states. This keeps the tabular methods manageable but does not represent every detail of the game (such as all enemies, rocks or the current room identity).

### Rewards and episode endings

| Event | Reward |
| --- | ---: |
| Enemy defeated | +1 |
| Room cleared | +2 |
| Heart lost | −3 |
| Player death | −10 |
| Exit reached in v1 | +5 |
| First visit to a new room in v2/v3 | +0.5 |
| Boss defeated in v2/v3 | +10 |
| Floor completed in v2/v3 | +20 |
| Each step in v1 | −0.01 |
| Each step in v2/v3 | −0.05 |

An episode ends on death, completion of the version's objective, or the step limit. Success means exiting the room in v1, completing the floor in v2, and completing **both floors** in v3.

## 2. Installation

**Recommended:** Python 3.13 and Git. The repository includes `mise.toml` for Python 3.13 and `requirements.txt` for the Python dependencies, including Gymnasium, NumPy, Pygame and TensorBoard.

```bash
git clone https://github.com/ruBBBBBBa/BasementCrawler.git
cd BasementCrawler

python -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt
python -m pip install -e .
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Activate the virtual environment again when opening a new terminal.

To check the installation:

```bash
python check_v1.py
python -m pytest -q
```

## 3. Play the game

Launch the graphical game with:

```bash
python play.py                # Final version: two random floors
python play.py Crawler-v2     # Fixed five-room floor
python play.py Crawler-v1     # Single room
```

| Key | Action |
| --- | --- |
| `W` / `A` / `S` / `D` | Move up / left / down / right |
| Arrow keys | Shoot in the corresponding direction |
| `Space` | Wait one turn |
| `R` | Restart |
| `Esc` | Quit |

Each key press advances the environment by one step. The top bar displays information such as remaining hearts, enemies and the shooting cooldown.

A text-based version of v1 is also available:

```bash
python play_text.py
```

## 4. Experiments and training

We trained SARSA, Expected SARSA and Q-Learning independently, using **three training seeds per algorithm and environment**. Every run starts from a zero-initialised Q-table; the 10-million-step v3 runs are **new training runs**, not continuations of the 2-million-step experiments.

### Experimental configuration

| Parameter | Value |
| --- | --- |
| Learning rate (`alpha`) | 0.1 |
| Discount factor (`gamma`) | 0.99 |
| Exploration policy | Epsilon-greedy |
| Initial / final epsilon | 1.0 / 0.05 |
| Epsilon decay | Linear over the first 70% of training |
| Training seeds | 3 per algorithm |
| Evaluation | 100 episodes per seed |
| Evaluation epsilon | 0.0 (greedy) |
| Evaluation seeds | Fixed, beginning at 10,000,000 |

### Main experiments

Run these commands from the project root. **Use different `--name` values if the output directories already exist**, to avoid overwriting the original experiments.

```bash
# Crawler-v1: 300,000 steps per algorithm and seed
python train.py --name v1_main --env Crawler-v1 --algo all --seeds 3 --steps 300000 --eval-every 10000

# Crawler-v2: 1,000,000 steps per algorithm and seed
python train.py --name v2_main --env Crawler-v2 --algo all --seeds 3 --steps 1000000 --eval-every 50000

# Crawler-v3: initial 2,000,000-step experiment
python train.py --name v3_main --env Crawler-v3 --algo all --seeds 3 --steps 2000000 --eval-every 200000

# Crawler-v3: independent extended 10,000,000-step experiment
python train.py --name v3_10m --env Crawler-v3 --algo all --seeds 3 --steps 10000000 --eval-every 500000
```

Use `python train.py --help` to inspect the additional options, including enemy movement probability, number of rooms, training algorithm and evaluation exploration.

### Final evaluation results

The following table shows the **mean final greedy success rate across three independent training seeds**. Each seed was evaluated over 100 episodes.

| Environment | SARSA | Expected SARSA | Q-Learning |
| --- | ---: | ---: | ---: |
| Crawler-v1 — 300k steps | 99.7% | 100.0% | 100.0% |
| Crawler-v2 — 1M steps | 85.7% | 91.3% | 84.7% |
| Crawler-v3 — 2M steps | 15.7% | 14.0% | 8.3% |
| Crawler-v3 — 10M steps | 25.0% | 37.0% | 9.3% |

All three methods learned v1 almost perfectly and performed well on v2. V3 was substantially harder: the random maps, multiple floors and frequent timeouts made successful navigation less reliable. After 10 million steps, Expected SARSA achieved the highest average success rate among the three algorithms. Results varied between training seeds, particularly on v3, so these findings should not be interpreted as proof of universal superiority.

**Important:** Mean episode returns should be compared **within** a version, not directly between v1, v2 and v3, because their objectives and reward structures differ.

## 5. TensorBoard logs and trained Q-tables

The training script writes the following outputs:

```text
runs/<experiment>/<algorithm>/seed<n>/
    TensorBoard event logs

qtables/<experiment>/<algorithm>_seed<n>.npy
    Final trained Q-table

qtables/<experiment>/<algorithm>_seed<n>.json
    Run configuration and final evaluation results
```

To inspect the learning curves:

```bash
tensorboard --logdir runs
```

Open <http://localhost:6006> in your browser. Important metrics include:

| TensorBoard tag | Meaning |
| --- | --- |
| `eval/success_rate` | Fraction of successful evaluation episodes |
| `eval/mean_return` | Average evaluation reward |
| `eval/death_rate` | Fraction of episodes ending in death |
| `eval/timeout_rate` | Fraction of episodes reaching the step limit |
| `eval/mean_rooms_visited` | Exploration progress in v2 |
| `eval/mean_floors_completed` | Progress through the two floors in v3 |
| `train/epsilon` | Exploration rate during training |
| `train/states_visited_pct` | Percentage of tabular observations visited |

The figures in the report display means over the three training seeds, individual seed curves and standard-deviation bands. They use mild smoothing for presentation; numerical results are taken from the final unsmoothed evaluations.

**Note:** TensorBoard logs and `.npy` Q-tables may be excluded from Git because they can be large. To reproduce an experiment, run the corresponding training command or use the saved checkpoints supplied separately with the submission.

## 6. Watch and record trained agents

`watch.py` loads a trained Q-table together with the adjacent JSON configuration, so the agent is replayed in the correct environment with the matching settings.

```bash
# Watch a saved Expected SARSA agent
python watch.py qtables/v3_10m/expected_sarsa_seed0.npy

# Replay and record a specific episode
python watch.py qtables/v3_10m/expected_sarsa_seed0.npy \
  --seed 10000000 --policy-seed 0 \
  --record videos/v3_demo_alternative --fps 8

# Search for a successful episode and record it
python watch.py qtables/v3_10m/expected_sarsa_seed0.npy \
  --find-win --pick shortest --search-limit 100 \
  --record videos/v3_demo --fps 8
```

When `--record` is used, the script saves MP4 recordings and a `recording_results.json` file summarising each episode. The `--seed` option controls the environment; `--policy-seed` controls random action tie-breaking. You can use `--random` for a baseline without a Q-table.

### Recorded gameplay demonstrations

The report includes four examples of Expected SARSA playing v3, showing both successful and unsuccessful behaviour:

| Training budget | Episode | Steps | Video |
| --- | --- | ---: | --- |
| 2M steps | Timeout after becoming stuck during navigation | 1,000 | [Watch](https://drive.google.com/file/d/1Ur2rUU9vofGABUZxQIUCXw9MVXVdRSDQ/view?usp=drive_link) |
| 2M steps | Successful completion of two floors | 118 | [Watch](https://drive.google.com/file/d/1KObdH3txqkn2MapzobyclHuR4GttoiSD/view?usp=drive_link) |
| 10M steps | Fast successful completion | 54 | [Watch](https://drive.google.com/file/d/1GT2MdSXM7EBkt_cSLjWbXVeFPuFqp1Nt/view?usp=drive_link) |
| 10M steps | Successful but less efficient behaviour | 100 | [Watch](https://drive.google.com/file/d/1laXfnEhSmJmOBo5UbzWtdhef-WZJt2SE/view?usp=drive_link) |

These recordings are **illustrative examples**, not a substitute for the quantitative evaluation across 100 episodes and three training seeds. The shared Drive files must be accessible to anyone with the link for the demonstrations to be viewable by the evaluator.

## 7. Use the environments from Python

```python
import gymnasium as gym
import crawler  # Registers Crawler-v1, Crawler-v2 and Crawler-v3

env = gym.make("Crawler-v3")
observation, info = env.reset(seed=0)

action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

The following rendering modes are supported:

- `render_mode="human"`: interactive Pygame window.
- `render_mode="rgb_array"`: image frames suitable for recording.
- `render_mode="ansi"`: text representation for debugging.

Training should be run without the graphical renderer for better performance.

## 8. Project structure

```text
BasementCrawler/
├── crawler/
│   ├── __init__.py          # Gymnasium registration for v1, v2 and v3
│   ├── crawler_v1.py        # Single-room environment
│   ├── crawler_v2.py        # Fixed five-room floor
│   ├── crawler_v3.py        # Procedurally generated two-floor environment
│   └── render.py            # Pygame graphics and video frames
├── tests/                   # Environment and mechanics tests
├── play.py                  # Graphical gameplay
├── play_text.py             # Text-mode gameplay for v1
├── check_v1.py              # Basic environment checks
├── train.py                 # Tabular algorithms, evaluation and logging
├── watch.py                 # Replay and record trained agents
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Package configuration
├── mise.toml                # Python version configuration
└── README.md
```

Generated folders such as `runs/`, `qtables/` and `videos/` are created when experiments and recordings are run; they may not be present in a fresh clone.

## 9. Main observations and limitations

The progression from v1 to v3 made it possible to examine how tabular agents behave as an environment becomes more difficult. In v2, the fixed room layout helps agents repeatedly encounter similar navigation situations. In v3, the map changes between episodes, and the agent must complete two floors within a fixed step limit.

The v3 recordings show that an agent may survive combat and even complete one floor, but later become stuck in a cleared room or take inefficient routes. The compact observation representation makes the Q-table feasible, although it does not capture every part of the game state. Most unsuccessful episodes in the extended v3 experiments ended in **timeout rather than death**.

The results demonstrate both the usefulness of tabular methods on manageable tasks and their limitations in more varied, procedurally generated environments. Further analysis is provided in the accompanying PDF report.

## 10. Reproducibility and submission notes

- The repository contains the environment implementation, training and evaluation scripts, graphical interface and tests.
- Exact training settings and final evaluations are saved beside the generated Q-tables in JSON files and recorded in TensorBoard.
- The main experiments use three independent training seeds. The 2M and 10M v3 experiments were trained independently from zero-initialised Q-tables.
- The precise version used for the submitted report should be identified by its Git commit hash (`git rev-parse HEAD`).
- Before submission, run `python check_v1.py`, `python -m pytest -q`, and check that the PDF video links are accessible.

**Assignment deadline:** 2 November 2026, 23:59 (Europe/Madrid).
