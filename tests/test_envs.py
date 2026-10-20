"""pytest -q   (run from the project folder)"""
import os
from collections import deque

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # headless pygame

import numpy as np
import gymnasium as gym
import pytest
from gymnasium.utils.env_checker import check_env

import crawler  # noqa: F401
from crawler.crawler_v1 import CrawlerV1Env, SIZE, START, DOOR_CELLS
from crawler.crawler_v2 import (CrawlerV2Env, ROOMS, EDGES, OPPOSITE, TRAPDOOR,
                                WALL, EXPLORED, UNEXPLORED_BRANCH, BOSS_NEXT_ROOM)

A, B, C, BOSS, S = (1, 2), (2, 1), (1, 3), (0, 3), (2, 2)   # see the map in crawler_v2.py


def put(env, room, cell, enemies=None):
    """Teleport the agent into `room` (marked visited); optionally replace its enemies."""
    env.visited.add(room)
    env._enter(room, cell)
    if enemies is not None:
        env.enemies = list(enemies)
        env.cleared = not enemies
    env.enemy_move_prob = 0.0
    return env


def rollout(env_id, seed, steps=300):
    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)
    rng, trace = np.random.default_rng(seed), [obs]
    for _ in range(steps):
        obs, r, term, trunc, info = env.step(int(rng.integers(9)))
        assert env.observation_space.contains(obs)
        trace.append((obs, r, term, trunc))
        if term or trunc:
            break
    return trace


# ------------------------------------------------------------------- shared
@pytest.mark.parametrize("env_id,n,limit", [("Crawler-v1", 5650, 200), ("Crawler-v2", 29675, 500)])
def test_api_spaces_and_time_limit(env_id, n, limit):
    env = gym.make(env_id)
    check_env(env.unwrapped)
    assert env.observation_space.n == n and env.action_space.n == 9
    env.reset(seed=0)
    env.unwrapped.enemies, env.unwrapped.hearts = [], 10**6      # nothing can end it early
    env.unwrapped.cleared = False                                # v1: door stays shut
    steps, trunc = 0, False
    while not trunc:
        _, _, term, trunc, _ = env.step(8)
        steps += 1
        assert not term
    assert steps == limit


@pytest.mark.parametrize("env_id", ["Crawler-v1", "Crawler-v2"])
def test_seeded_rollouts_repeat_and_stay_in_bounds(env_id):
    for seed in range(30):
        assert rollout(env_id, seed) == rollout(env_id, seed)


@pytest.mark.parametrize("env_cls", [CrawlerV1Env, CrawlerV2Env])
def test_render_modes_do_not_change_state(env_cls):
    for mode in ("ansi", "rgb_array", "human"):
        env = env_cls(render_mode=mode)
        env.reset(seed=3)
        for _ in range(5):
            env.step(4)
        before = (env.agent, list(env.enemies), env.hearts, env.cooldown, env.tear,
                  env.np_random.bit_generator.state["state"]["state"])
        frame = env.render()
        frame2 = env.render()
        after = (env.agent, list(env.enemies), env.hearts, env.cooldown, env.tear,
                 env.np_random.bit_generator.state["state"]["state"])
        assert before == after
        if mode == "rgb_array":
            assert frame.dtype == np.uint8 and frame.shape == (504, 448, 3)
            assert np.array_equal(frame, frame2)
        elif mode == "ansi":
            assert isinstance(frame, str) and "A" in frame
        env.close()


# ----------------------------------------------------------------------- v1
def test_v1_spawn_constraints():
    env = CrawlerV1Env(n_enemies=3)
    for seed in range(200):
        env.reset(seed=seed)
        assert len(set(env.enemies)) == 3
        assert all(abs(r - START[0]) + abs(c - START[1]) >= 3 for r, c in env.enemies)


# ----------------------------------------------------------------------- v2
def test_v2_doors_are_reciprocal_and_tree():
    env = CrawlerV2Env()
    assert len(EDGES) == len(ROOMS) - 1                       # connected + n-1 edges = tree
    for room, doors in env.doors.items():
        for d, nxt in doors.items():
            assert env.doors[nxt][OPPOSITE[d]] == room
    seen, todo = {S}, [S]
    while todo:
        for nxt in env.doors[todo.pop()].values():
            if nxt not in seen:
                seen.add(nxt)
                todo.append(nxt)
    assert seen == set(ROOMS)


def test_v2_rocks_never_trap_the_agent():
    for room, (_, rocks) in ROOMS.items():
        free = {(r, c) for r in range(SIZE) for c in range(SIZE)} - set(rocks)
        assert set(DOOR_CELLS.values()) | {START} | ({TRAPDOOR} if room == BOSS else set()) <= free
        reach, todo = {START}, deque([START])
        while todo:
            r, c = todo.popleft()
            for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nb in free and nb not in reach:
                    reach.add(nb)
                    todo.append(nb)
        assert reach == free, room


def test_v2_room_transition_entry_reward_and_locked_doors():
    env = CrawlerV2Env()
    env.reset(seed=0)
    env.agent, env.hearts = DOOR_CELLS[0], 2
    _, r, term, _, info = env.step(0)                          # north door of the start room
    assert env.room == A and env.agent == DOOR_CELLS[1] and not term
    assert r == pytest.approx(-0.01 + 0.5)
    assert info["hearts"] == 2 and info["rooms_visited"] == 2
    assert len(env.enemies) == 2 and env.tear is None
    for e in env.enemies:
        assert e not in env.rocks and abs(e[0] - 4) + abs(e[1] - 2) >= 3
    env.enemy_move_prob = 0.0
    env.step(1)                                                # back south: locked
    assert env.room == A


def test_v2_revisit_gives_no_entry_reward_and_room_stays_clear():
    env = CrawlerV2Env()
    env.reset(seed=0)
    put(env, A, DOOR_CELLS[1], enemies=[])
    env.cleared_rooms.add(A)
    _, r, *_ = env.step(1)                                     # A -> start (visited)
    assert env.room == S and r == pytest.approx(-0.01)
    env.agent = DOOR_CELLS[0]
    _, r, *_ = env.step(0)                                     # start -> A again
    assert env.room == A and env.cleared and env.enemies == [] and r == pytest.approx(-0.01)


def test_v2_rocks_block_player_tears_and_enemies():
    env = CrawlerV2Env()
    env.reset(seed=0)
    put(env, A, (2, 1), enemies=[(0, 1)])                      # rock at (1, 1) in between
    env.step(0)
    assert env.agent == (2, 1)                                 # player blocked
    env.step(4)
    env.step(8)
    assert env.enemies == [(0, 1)] and env.tear is None        # tear stopped by the rock
    env.enemy_move_prob = 1.0
    env.step(8)
    assert env.enemies == [(0, 1)]                             # chaser blocked by the rock too


def test_v2_boss_takes_three_hits_then_trapdoor_completes_floor():
    env = CrawlerV2Env()
    env.reset(seed=0)
    put(env, BOSS, (4, 2), enemies=[(2, 2)])
    env.boss_hp = 3
    rewards = []
    for _ in range(3):
        _, r, *_ = env.step(4)                                 # boss two cells north
        rewards.append(r)
        env.cooldown = 0
    assert env.boss_defeated and env.enemies == [] and env.cleared
    assert rewards[:2] == [pytest.approx(-0.01)] * 2
    assert rewards[2] == pytest.approx(-0.01 + 10 + 2)
    _, r, term, _, info = env.step(0)                          # (3, 2)
    assert not term
    _, r, term, _, info = env.step(0)                          # onto the trapdoor
    assert term and info["floor_completed"] and info["boss_defeated"]
    assert r == pytest.approx(-0.01 + 20)


def test_v2_trapdoor_closed_while_boss_alive():
    env = CrawlerV2Env()
    env.reset(seed=0)
    put(env, BOSS, (3, 2), enemies=[(0, 0)])
    env.boss_hp = 3
    _, _, term, _, info = env.step(0)
    assert env.agent == TRAPDOOR and not term and not info["floor_completed"]
    obs = env._get_obs()
    assert obs < env.combat_n                                  # still combat mode


def test_v2_boss_room_observation_uses_boss_hp():
    env = CrawlerV2Env()
    env.reset(seed=0)
    put(env, BOSS, (4, 2), enemies=[(0, 0)])
    obs = []
    for hp in (3, 2, 1):
        env.boss_hp = hp
        obs.append(env._get_obs())
    assert len(set(obs)) == 3


def test_v2_branch_door_labels():
    env = CrawlerV2Env()
    env.reset(seed=0)
    assert [env.door_label(d) for d in range(4)] == [UNEXPLORED_BRANCH, WALL, UNEXPLORED_BRANCH, WALL]
    env.visited |= {B}
    assert env.door_label(2) == EXPLORED                       # dead end done
    env.visited |= {A}
    assert env.door_label(0) == UNEXPLORED_BRANCH              # C and boss still behind A
    put(env, A, DOOR_CELLS[1], enemies=[])
    assert env.door_label(3) == UNEXPLORED_BRANCH and env.door_label(1) == EXPLORED
    put(env, C, DOOR_CELLS[2], enemies=[])
    assert env.door_label(0) == BOSS_NEXT_ROOM                 # boss beats unexplored
    assert env.door_label(2) == EXPLORED


def test_v2_dead_and_navigation_observations_in_bounds():
    env = CrawlerV2Env()
    env.reset(seed=0)
    put(env, A, (2, 2), enemies=[(1, 2)])
    env.hearts = 1
    env.enemy_move_prob = 1.0
    obs, r, term, _, info = env.step(8)                        # enemy bumps: last heart
    assert term and info["dead"] and env.observation_space.contains(obs)
    env.reset(seed=0)
    put(env, BOSS, (4, 2), enemies=[])
    env.boss_defeated = True
    obs = env._get_obs()
    assert env.combat_n <= obs < env.observation_space.n
