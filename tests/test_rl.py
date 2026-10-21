import json
import os

import numpy as np
import gymnasium as gym
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

import crawler  # noqa: F401
from crawler.crawler_v1 import CrawlerV1Env
from crawler.crawler_v2 import CrawlerV2Env, WALL
from crawler.scenarios import SCENARIOS
from crawler.tabular import expected_value, greedy, linear_epsilon, td_target
from crawler.transfer import v1_to_v2_q
from evaluate import EVAL_SEED_BASE, evaluate
from train import TRAIN_SEED_MAX, train


# ------------------------------------------------------------ update rules
Q = np.array([[0.0, 0.0, 0.0],
              [1.0, 3.0, 3.0]])          # state 1 has a tie between actions 1 and 2


def test_td_targets():
    g, r = 0.5, 2.0
    assert td_target("sarsa", Q, r, 1, 0, 0.3, g, False) == r + g * 1.0
    assert td_target("q_learning", Q, r, 1, 0, 0.3, g, False) == r + g * 3.0
    # eps=0.3: every action 0.1, the two tied maxima share the other 0.7 -> 0.45 each
    assert expected_value(Q[1], 0.3) == pytest.approx(0.1 * 1 + 0.45 * 3 + 0.45 * 3)
    assert td_target("expected_sarsa", Q, r, 1, 0, 0.3, g, False) == pytest.approx(r + g * 2.8)
    assert expected_value(Q[1], 0.0) == pytest.approx(3.0)
    for algo in ("sarsa", "expected_sarsa", "q_learning"):
        assert td_target(algo, Q, r, 1, 0, 0.3, g, True) == r   # terminal: no bootstrap
    with pytest.raises(ValueError):
        td_target("dqn", Q, r, 1, 0, 0.3, g, False)


def test_greedy_ties_are_random_but_seeded():
    picks = [greedy(Q[1], np.random.default_rng(s)) for s in range(50)]
    assert set(picks) == {1, 2}
    assert picks == [greedy(Q[1], np.random.default_rng(s)) for s in range(50)]


def test_linear_epsilon():
    assert linear_epsilon(0, 1000) == 1.0
    assert linear_epsilon(700, 1000) == pytest.approx(0.05)
    assert linear_epsilon(350, 1000) == pytest.approx(0.525)
    assert linear_epsilon(999, 1000) == pytest.approx(0.05)


# ------------------------------------------------------- truncation handling
class OneStep(gym.Env):
    """One state, one action, reward 1. Never terminates; TimeLimit truncates every step."""
    observation_space, action_space = spaces.Discrete(1), spaces.Discrete(1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return 0, {"hearts": 1}

    def step(self, action):
        return 0, 1.0, False, False, {"hearts": 1}


gym.register("OneStepTest-v0", entry_point=OneStep, max_episode_steps=1)


@pytest.mark.parametrize("algo", ["sarsa", "expected_sarsa", "q_learning"])
def test_truncation_bootstraps(algo):
    q, cfg = train("OneStepTest-v0", algo=algo, steps=2000, gamma=0.5, alpha=0.5)
    assert cfg["episodes"] == 2000
    assert q[0, 0] == pytest.approx(1 / (1 - 0.5))           # 1 if it did not bootstrap


# ------------------------------------------------------------- full pipeline
def test_train_is_reproducible_and_saves_a_checkpoint(tmp_path):
    out = str(tmp_path / "run")
    q1, cfg = train("Crawler-v1", algo="sarsa", steps=3000, seed=4, out=out)
    q2, _ = train("Crawler-v1", algo="sarsa", steps=3000, seed=4, tb=False)
    q3, _ = train("Crawler-v1", algo="sarsa", steps=3000, seed=5, tb=False)
    assert np.array_equal(q1, q2) and not np.array_equal(q1, q3)
    assert np.array_equal(np.load(os.path.join(out, "q.npy")), q1)
    saved = json.load(open(os.path.join(out, "config.json")))
    assert saved["obs_n"] == 5650 and saved["action_n"] == 9 and saved["seed"] == 4
    assert any(f.startswith("events") for f in os.listdir(out))
    assert TRAIN_SEED_MAX <= EVAL_SEED_BASE


def test_evaluate_is_repeatable_and_leaves_q_alone():
    q, _ = train("Crawler-v1", steps=3000, seed=1, tb=False)
    before = q.copy()
    s1, rows1 = evaluate(q, "Crawler-v1", episodes=20)
    s2, rows2 = evaluate(q, "Crawler-v1", episodes=20)
    assert rows1 == rows2 and np.array_equal(q, before) and q.flags.writeable
    assert rows1[0]["seed"] == EVAL_SEED_BASE
    e1, _ = evaluate(q, "Crawler-v1", episodes=20, eps=0.5)
    e2, _ = evaluate(q, "Crawler-v1", episodes=20, eps=0.5)
    assert e1 == e2 and np.array_equal(q, before)


def test_init_from_continues_budget(tmp_path):
    a = str(tmp_path / "a")
    train("Crawler-v2", scenario="boss_only", steps=1000, out=a, tb=False)
    q, cfg = train("Crawler-v2", steps=1000, init_from=a, tb=False)
    assert cfg["init"] == "copy" and cfg["total_steps"] == 2000 and cfg["parent_steps"] == 1000


# ---------------------------------------------------------------- transfer
def matched_states(rng, n=400):
    """Random combat situations as (agent, enemies, cooldown, hearts)."""
    cells = [(r, c) for r in range(5) for c in range(5)]
    for _ in range(n):
        k = int(rng.integers(1, 4))
        idx = rng.choice(25, size=k + 1, replace=False)
        yield cells[idx[0]], [cells[i] for i in idx[1:]], int(rng.integers(3)), int(rng.integers(1, 4))


def test_v1_to_v2_mapping_matches_game_states():
    v1, v2 = CrawlerV1Env(), CrawlerV2Env()
    v1.reset(seed=0)
    v2.reset(seed=0)
    v2.room = (1, 2)                                          # an ordinary (non-boss) room
    q1 = np.arange(v1.observation_space.n * 9, dtype=float).reshape(-1, 9)   # unique rows
    q2 = v1_to_v2_q(q1, v1, v2)
    assert q2.shape == (29675, 9)
    for agent, enemies, cooldown, hearts in matched_states(np.random.default_rng(0)):
        for env in (v1, v2):
            env.agent, env.enemies, env.cooldown = agent, list(enemies), cooldown
        v2.hearts = hearts
        o1, o2 = v1._get_obs(), v2._get_obs()
        assert o1 < v1.combat_n and o2 < v2.combat_n
        assert np.array_equal(q2[o2], q1[o1])                 # same situation, same values
    assert not q2[v2.combat_n:].any()                         # navigation rows left at zero


def test_v1_to_v2_rejects_other_encodings():
    v1, v2 = CrawlerV1Env(fire_delay=3), CrawlerV2Env()
    with pytest.raises(ValueError):
        v1_to_v2_q(np.zeros((v1.observation_space.n, 9)), v1, v2)


# --------------------------------------------------------------- curriculum
@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenarios_keep_the_v2_encoding(name):
    env = gym.make("Crawler-v2", **SCENARIOS[name])
    check_env(env.unwrapped)
    assert env.observation_space.n == 29675 and env.action_space.n == 9
    u = env.unwrapped
    u.reset(seed=0)
    assert u.room == u.start_room and u.boss_room in u.rooms
    for room in u.rooms:                                      # doors out of the scenario are walls
        assert set(u.doors[room].values()) <= set(u.rooms)
    if name == "hard_boss":
        u.enemies, u.cleared = [], True
        assert u.door_label(2) == WALL                       # the door back to medium is gone
