"""Tests for Crawler-v3: floor generator, room pools, floors, boss heal, compact labels."""
from collections import Counter, defaultdict

import numpy as np
import pytest
import gymnasium as gym

import crawler  # noqa: F401
from crawler.crawler_v1 import DIRS
from crawler.crawler_v2 import CrawlerV2Env, TRAPDOOR
from crawler.crawler_v3 import (CrawlerV3Env, generate_floor, START_ROOM, EASY, MEDIUM,
                                HARD, POOLS_BY_FLOOR)


def _tree_distances(edges):
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    dist, stack = {START_ROOM: 0}, [START_ROOM]
    while stack:
        room = stack.pop()
        for other in adj[room]:
            if other not in dist:
                dist[other] = dist[room] + 1
                stack.append(other)
    return dist


@pytest.mark.parametrize("n_rooms", [4, 6, 8, 10])
def test_generator_builds_connected_trees_with_boss_in_a_dead_end(n_rooms):
    for seed in range(200):
        rooms, edges, boss = generate_floor(np.random.default_rng(seed), n_rooms)
        assert len(rooms) == n_rooms and len(edges) == n_rooms - 1      # a tree
        assert set(_tree_distances(edges)) == rooms                    # connected
        for a, b in edges:                                             # doors join neighbours
            assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
        degree = Counter(r for e in edges for r in e)
        assert degree[boss] == 1 and boss != START_ROOM                # boss is a dead end
        assert _tree_distances(edges)[boss] >= 2                       # never next to the start


def test_same_seed_same_map_and_different_seeds_differ():
    a, b = CrawlerV3Env(), CrawlerV3Env()
    a.reset(seed=7)
    b.reset(seed=7)
    assert a.rooms == b.rooms and a.boss_room == b.boss_room
    maps = set()
    for seed in range(20):
        a.reset(seed=seed)
        maps.add(tuple(sorted(a.rooms)))
    assert len(maps) > 5


def test_map_seed_fixes_the_maps():
    env = CrawlerV3Env(map_seed=0)
    env.reset(seed=1)
    first = dict(env.rooms)
    env.reset(seed=2)
    assert env.rooms == first


def test_rooms_come_from_the_floor_pools():
    env = CrawlerV3Env(n_rooms=(8, 8))
    for seed in range(30):
        env.reset(seed=seed)
        for room, content in env.rooms.items():
            if room == START_ROOM:
                assert content == (0, ())
            elif room == env.boss_room:
                assert content[0] == 1 and TRAPDOOR not in content[1]
            else:
                assert content in POOLS_BY_FLOOR[0]
                assert content not in HARD
    assert set(POOLS_BY_FLOOR[1]) == set(MEDIUM + HARD) and not set(POOLS_BY_FLOOR[1]) & set(EASY)


def test_rocks_never_block_doors_or_split_a_room():
    for _, rocks in EASY + MEDIUM + HARD:
        free = {(r, c) for r in range(5) for c in range(5)} - set(rocks)
        assert {(0, 2), (4, 2), (2, 0), (2, 4)} <= free
        seen, stack = {(4, 2)}, [(4, 2)]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                nb = (r + dr, c + dc)
                if nb in free and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        assert seen == free


def _stand_next_to_open_trapdoor(env):
    env.room, env.enemies, env.cleared = env.boss_room, [], True
    env.rocks = frozenset()
    env.boss_defeated, env.agent = True, (1, 2)


def test_trapdoor_on_floor_1_leads_to_floor_2_and_keeps_hearts():
    env = CrawlerV3Env()
    env.reset(seed=3)
    floor1 = dict(env.rooms)
    env.hearts = 2
    _stand_next_to_open_trapdoor(env)
    obs, r, terminated, truncated, info = env.step(1)        # step south onto the trapdoor
    assert not terminated and r == pytest.approx(env.step_reward + 20)
    assert info["floor"] == 2 and info["floors_completed"] == 1 and not info["run_completed"]
    assert env.room == START_ROOM and env.hearts == 2 and env.visited == {START_ROOM}
    assert env.rooms != floor1 and not env.floor_completed
    assert all(content in POOLS_BY_FLOOR[1] for room, content in env.rooms.items()
               if room not in (START_ROOM, env.boss_room))


def test_trapdoor_on_last_floor_ends_the_run():
    env = CrawlerV3Env()
    env.reset(seed=3)
    _stand_next_to_open_trapdoor(env)
    env.step(1)
    _stand_next_to_open_trapdoor(env)
    obs, r, terminated, truncated, info = env.step(1)
    assert terminated and info["run_completed"] and info["floors_completed"] == 2
    assert info["floor_completed"]                    # scripts that check this still work


def test_killing_the_boss_restores_one_heart():
    env = CrawlerV3Env()
    env.reset(seed=4)
    env.hearts = 1
    env._enter(env.boss_room, (4, 2))
    env.boss_hp = 1
    env._hit(env.enemies[0])
    assert env.boss_defeated and env.hearts == 2
    env.hearts = 3
    env.boss_defeated, env.enemies, env.boss_hp = False, [(0, 0)], 1
    env._hit((0, 0))
    assert env.hearts == 3                            # never above the maximum


def test_compact_navigation_merges_wall_and_explored():
    assert CrawlerV3Env().observation_space.n == 16_875 + 25 * 3**4 * 2      # 20,925
    assert CrawlerV3Env(compact_nav=False).observation_space == CrawlerV2Env().observation_space
    env = CrawlerV3Env()
    env.reset(seed=0)
    env.enemies, env.cleared = [], True
    labels = [env.door_label(d) for d in range(4)]
    cell, *compact, trap = np.unravel_index(env._get_obs() - env.combat_n, env.nav_dims)
    assert compact == [{2: 1, 3: 2}.get(lab, 0) for lab in labels] and trap == 0


def test_room_counts_per_floor():
    assert CrawlerV3Env().n_rooms == (4, 5)
    assert CrawlerV3Env(n_rooms=6).n_rooms == (6, 6)
    assert CrawlerV3Env(n_rooms=(4, 5), n_floors=3).n_rooms == (4, 5, 5)
    env = CrawlerV3Env(n_rooms=(4, 6))
    env.reset(seed=0)
    assert len(env.rooms) == 4


def test_random_play_runs():
    env = gym.make("Crawler-v3")
    obs, _ = env.reset(seed=0)
    for _ in range(2000):
        assert env.observation_space.contains(obs)
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, _ = env.reset()
