"""Basement Crawler v3

Everything from v2 (rooms, locked doors, rocks, hearts, boss, trapdoor, door labels),
plus:
- a new random floor every episode, built with The Binding of Isaac's floor generator
  on a 5x5 grid: rooms grow outward from the start room, never forming loops, and the
  boss sits in the last dead end found (one of the rooms furthest from the start);
- room contents drawn from easy / medium / hard pools: floor 1 uses easy and medium
  rooms, floor 2 uses medium and hard ones (as Isaac does within a chapter);
- two floors: the trapdoor on floor 1 drops the agent into floor 2, and the trapdoor on
  the last floor ends the run. Killing a boss restores one heart.

Observation: v2's combat part, and a compact navigation part. On a tree-shaped floor a
door marked EXPLORED is never worth taking, exactly like a wall, so v3 merges the two
labels: each door is "closed" (wall or explored), "unexplored" or "boss". That cuts the
navigation states from 12,800 to 4,050 (20,925 states in total); compact_nav=False
restores v2's encoding (29,675). The observation never says which room or floor the agent
is in, which is why one Q-table can play maps it has never seen.
"""
import numpy as np
from gymnasium import spaces

from .crawler_v1 import CrawlerV1Env, DIRS, START
from .crawler_v2 import CrawlerV2Env, build_doors, UNEXPLORED_BRANCH, BOSS_NEXT_ROOM

SIZE_FLOOR = 5                         # the floor is a 5x5 grid of possible rooms
START_ROOM = (2, 2)                    # the start room is the centre of that grid

# Room pools: (number of enemies, rock cells). Rocks never sit on a door cell, and every
# free cell stays reachable.
EASY = [(1, ()), (1, ((1, 1), (3, 3))), (1, ((1, 3), (3, 1)))]
MEDIUM = [(2, ((1, 1), (1, 3), (3, 1), (3, 3))), (2, ((2, 1), (2, 3))), (2, ())]
HARD = [(3, ((2, 2),)), (3, ((1, 2), (3, 2))), (3, ((1, 1), (3, 3)))]
BOSS_ROCKS = [((1, 1), (1, 3)), ((3, 1), (3, 3)), ()]   # centre stays free for the trapdoor
POOLS_BY_FLOOR = [EASY + MEDIUM, MEDIUM + HARD]


def generate_floor(rng, n_rooms, max_tries=1000):
    """Isaac's floor generator, scaled to a 5x5 grid. Returns (rooms, edges, boss_room).

    Breadth-first growth from the start room. A neighbour cell is skipped if it is
    outside the grid or already a room, if it already touches more than one room (this
    keeps the floor a tree), if the floor is full, or on a 50% coin flip. A room that
    adds no neighbour is a dead end; the boss goes in the last dead end recorded.
    The floor is rejected and regrown if it has the wrong size or the boss would be
    next to the start room.
    """
    for _ in range(max_tries):
        rooms, edges, queue, dead_ends = {START_ROOM}, [], [START_ROOM], []
        while queue:
            cell = queue.pop(0)
            added = False
            for dr, dc in DIRS:
                nb = (cell[0] + dr, cell[1] + dc)
                if not (0 <= nb[0] < SIZE_FLOOR and 0 <= nb[1] < SIZE_FLOOR) or nb in rooms:
                    continue
                touching = sum((nb[0] + r, nb[1] + c) in rooms for r, c in DIRS)
                if touching > 1 or len(rooms) >= n_rooms or rng.random() < 0.5:
                    continue
                rooms.add(nb)
                edges.append((cell, nb))
                queue.append(nb)
                added = True
            if not added and cell != START_ROOM:
                dead_ends.append(cell)
        if len(rooms) != n_rooms or not dead_ends:
            continue
        boss = dead_ends[-1]
        if abs(boss[0] - START_ROOM[0]) + abs(boss[1] - START_ROOM[1]) == 1:
            continue
        return rooms, edges, boss
    raise RuntimeError(f"could not generate a floor with {n_rooms} rooms")


class CrawlerV3Env(CrawlerV2Env):
    def __init__(self, n_rooms=(4, 5), n_floors=2, boss_heal=1, map_seed=None,
                 compact_nav=True, **kwargs):
        super().__init__(**kwargs)
        self.compact_nav = compact_nav
        if compact_nav:      # door labels: 0 closed (wall/explored), 1 unexplored, 2 boss
            self.nav_dims = (25, 3, 3, 3, 3, 2)
            self.observation_space = spaces.Discrete(self.combat_n + int(np.prod(self.nav_dims)))
        self.n_floors = n_floors
        sizes = tuple(int(n) for n in np.atleast_1d(n_rooms))   # one per floor; the last repeats
        self.n_rooms = tuple(sizes[min(i, len(sizes) - 1)] for i in range(n_floors))
        assert all(3 <= n <= 10 for n in self.n_rooms), "floors need 3-10 rooms"
        self.boss_heal = boss_heal
        self.map_seed = map_seed     # set it to play the same maps every episode (debugging)

    def reset(self, seed=None, options=None):
        super(CrawlerV1Env, self).reset(seed=seed)     # seed self.np_random first
        self.floor, self.run_completed = 0, False
        self._build_floor()
        return super().reset(seed=None, options=options)   # v2 reset: start room, hearts...

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.floor_completed and self.floor + 1 < self.n_floors:
            self.floor += 1                # trapdoor: drop into the next floor
            self._build_floor()
            self._start_floor()
            terminated = False
            self._render_human()
            obs, info = self._get_obs(), self._info()
        elif self.floor_completed:
            self.run_completed = True
            info = self._info()
        return obs, reward, terminated, truncated, info

    # NEW MECHANICS
    def _build_floor(self):
        #Generate the current floor and fill its rooms from the floor's pools.
        rng = (np.random.default_rng(self.map_seed + self.floor) if self.map_seed is not None
               else self.np_random)
        cells, edges, boss = generate_floor(rng, self.n_rooms[self.floor])
        pool = POOLS_BY_FLOOR[min(self.floor, len(POOLS_BY_FLOOR) - 1)]
        rooms = {}
        for cell in sorted(cells):
            if cell == START_ROOM:
                rooms[cell] = (0, ())
            elif cell == boss:
                rooms[cell] = (1, BOSS_ROCKS[int(rng.integers(len(BOSS_ROCKS)))])
            else:
                rooms[cell] = pool[int(rng.integers(len(pool)))]
        self.rooms, self.boss_room, self.start_room = rooms, boss, START_ROOM
        self.doors = build_doors(rooms, edges)

    def _start_floor(self):
        #Same as v2's reset, but hearts carry over from the previous floor.
        self.agent, self.cooldown, self.tear = START, 0, None
        self.room, self.visited, self.cleared_rooms = self.start_room, {self.start_room}, set()
        self.boss_hp, self.boss_defeated, self.floor_completed = 0, False, False
        self.room_changed = False
        self._enter(self.start_room, START)

    def _hit(self, cell):
        reward = super()._hit(cell)
        if self.room == self.boss_room and self.boss_defeated:
            self.hearts = min(self.max_hearts, self.hearts + self.boss_heal)  # boss drops a heart
        return reward

    # OBSERVATION AND INFO
    def _get_obs(self):
        if self.enemies or not self.compact_nav:
            return super()._get_obs()                  # combat: identical to v2
        compact = {UNEXPLORED_BRANCH: 1, BOSS_NEXT_ROOM: 2}  # wall and explored -> 0
        labels = [compact.get(self.door_label(d), 0) for d in range(4)]
        cell = self.agent[0] * 5 + self.agent[1]
        trapdoor = int(self.trapdoor_open and self.room == self.boss_room)
        return self.combat_n + int(np.ravel_multi_index((cell, *labels, trapdoor), self.nav_dims))

    def _info(self):
        info = super()._info()
        info.update(floor=self.floor + 1, n_floors=self.n_floors,
                    floors_completed=self.floor + int(self.floor_completed),
                    run_completed=self.run_completed)
        return info

    def _render_text(self):
        return super()._render_text() + f"  floor {self.floor + 1}/{self.n_floors}"
