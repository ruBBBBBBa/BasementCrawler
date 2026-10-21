"""Basement Crawler v2

Room mechanics (moves, tears, chasers, step order) are inherited from v1. 
This version adds:
- the floor as a tree of rooms joined by doors; doors stay locked while enemies live;
- rocks block the player, enemies and tears;
- hearts carry over between rooms;
- the boss room holds one boss with 3 HP; killing it opens a trapdoor in the room
  centre, and stepping onto it completes the floor.

Observation:
- combat (enemies alive): 
    cell x nearest-enemy offset x cooldown x enemies left (boss HP in the boss room) x hearts
- navigation (room clear): 
    cell x a label per door x trapdoor open 
"""
import numpy as np
from gymnasium import spaces

from .crawler_v1 import (CrawlerV1Env, DOOR_CELLS, DIRS, SIZE, START,
                         R_CLEAR, R_DEATH, R_STEP)

#Rewards for v2
R_ENTER, R_BOSS, R_FLOOR = 0.5, 10.0, 20.0

OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}
TRAPDOOR = (2, 2)
WALL, EXPLORED, UNEXPLORED_BRANCH, BOSS_NEXT_ROOM = range(4)   # door labels

# Rooms sit on a 5x5 floor grid (row, col) that represents the layout of the floor. 
# Each room: (enemies, rocks).
START_ROOM, BOSS_ROOM = (2, 2), (0, 3)
ROOMS = {
    (2, 2): (0, ()),    # start, empty room --> [S]
    (2, 1): (1, ((1, 1), (3, 3))),  # easy room: dead end --> [W]
    (1, 2): (2, ((1, 1), (1, 3), (3, 1), (3, 3))),  # medium room: four rocks, two enemies --> [A]
    (1, 3): (3, ((2, 2),)), # hard room: centre rock, three enemies --> [C]
    (0, 3): (1, ((1, 1), (1, 3))),    # boss room: two rocks, one enemy (the boss) --> [B]
}
#LAYOUT (still a fixed floor, yet to implement the random floor generator):

#   . . . B . 
#   . . A C .
#   . W S . .

#The edges gives pairs of room coordinates that share a door.
EDGES = [((2, 2), (1, 2)), ((2, 2), (2, 1)), ((1, 2), (1, 3)), ((1, 3), (0, 3))]


def build_doors(rooms, edges):
    #converts undirected edges to a dictionary mapping each room’s direction to its neighboring room
    #OPPOSITE maps N to S and W to E
    doors = {room: {} for room in rooms}
    for a, b in edges:
        d = DIRS.index((b[0] - a[0], b[1] - a[1]))   # rooms must be grid neighbours
        doors[a][d], doors[b][OPPOSITE[d]] = b, a
    return doors

#The second version inherits from the first one
class CrawlerV2Env(CrawlerV1Env):
    def __init__(self, hearts=3, enemy_move_prob=0.5, enter_reward=R_ENTER,
                 rooms=ROOMS, edges=EDGES, start_room=START_ROOM, boss_room=BOSS_ROOM,
                 render_mode=None, **kwargs):
        assert 1 <= hearts <= 3, "v2 observation encodes 1-3 hearts"
        super().__init__(n_enemies=1, enemy_move_prob=enemy_move_prob, hearts=hearts,
                         render_mode=render_mode, **kwargs)
        self.enter_reward = enter_reward
        self.rooms, self.start_room, self.boss_room = rooms, start_room, boss_room
        self.doors = build_doors(rooms, edges)
        self.combat_dims = (SIZE * SIZE, 25, self.fire_delay + 1, 3, 3)
        self.combat_n = int(np.prod(self.combat_dims))
        self.nav_dims = (SIZE * SIZE, 4, 4, 4, 4, 2)
        #rebuilds the observation space to include the navigation states (once the room is clear)
        self.observation_space = spaces.Discrete(self.combat_n + int(np.prod(self.nav_dims)))

    #initialized room, visited, cleared, boss_hp, boss_defeated, floor_completed, room_changed before entering the start room
    #important difference: visited tracks exploration and the one-time enter reward, cleared tracks which rooms have been cleared
    def reset(self, seed=None, options=None):
        super(CrawlerV1Env, self).reset(seed=seed)  # seeds self.np_random
        self.agent, self.hearts, self.cooldown = START, self.max_hearts, 0
        self.tear, self.exited = None, False
        self.room, self.visited, self.cleared_rooms = self.start_room, {self.start_room}, set()
        self.boss_hp, self.boss_defeated, self.floor_completed = 0, False, False
        self.room_changed = False
        self._enter(self.start_room, START)
        self._render_human()
        return self._get_obs(), self._info()

    #Main controller of the game (v2)
    def step(self, action):
        action, reward = int(action), R_STEP
        self.room_changed = False
        if self.cooldown > 0:
            self.cooldown -= 1
        if action < 4:
            reward += self._move_agent(action)
        elif action < 8 and self.cooldown == 0 and self.tear is None:
            self.tear = [self.agent, action - 4, self.tear_range]
            self.cooldown = self.fire_delay
        if self.floor_completed:
            self._render_human()
            return self._get_obs(), reward + R_FLOOR, True, False, self._info()
        if not self.room_changed:
            reward += self._advance_tear()
            reward += self._move_enemies()
        if not self.enemies and not self.cleared:
            self.cleared = True
            self.cleared_rooms.add(self.room)
            reward += R_CLEAR
        terminated = self.hearts <= 0
        if terminated:
            reward += R_DEATH
        self._render_human()
        return self._get_obs(), reward, terminated, False, self._info()



#NEW MECHANICS:

    def _enter(self, room, cell):
        #Puts the agent in `room` at `cell`
        #spawn enemies on the first visit. Once a room is cleared, it stays cleared (no enemy respawn)
        self.room, self.agent, self.tear = room, cell, None
        n, rocks = self.rooms[room]
        self.rocks = frozenset(rocks)
        self.enemies = []
        if room not in self.cleared_rooms and n > 0:
            free = [(r, c) for r in range(SIZE) for c in range(SIZE)
                    if (r, c) not in self.rocks and abs(r - cell[0]) + abs(c - cell[1]) >= 3]
            idx = self.np_random.choice(len(free), size=n, replace=False)
            self.enemies = [free[i] for i in idx]
            #initializes the boss HP if the room is the boss room
            if room == self.boss_room:
                self.boss_hp = 3
        if not self.enemies:
            self.cleared_rooms.add(room)
        self.cleared = room in self.cleared_rooms

    def _use_door(self, d):
        #looks up the next room behind door d
        nxt = self.doors[self.room].get(d)
        if nxt is None:
            return 0.0  # plain wall
        #it checks whether is the first visit
        first = nxt not in self.visited
        self.visited.add(nxt)
        #enters the next room through the door cell; entering from North puts the player in the South cell of the next room (for example)
        self._enter(nxt, DOOR_CELLS[OPPOSITE[d]])
        self.room_changed = True
        #gives the reward if it's the first visit
        return self.enter_reward if first else 0.0
    #a room change step intentionally skips the tear and enemy moves, the new room begins calmly for one turn

#OLD MECHANICS (inherited from v1, but some methods are overridden):
    def _move_agent(self, d):
        before = self.agent
        reward = super()._move_agent(d)
        if (self.trapdoor_open and self.agent == TRAPDOOR != before
                and self.room == self.boss_room):
            self.floor_completed = True
        return reward

    #added the rocks collision
    def _blocked(self, cell):
        return cell in self.rocks 

    #added the boss hit logic
    def _hit(self, cell):
        if self.room != self.boss_room:
            return super()._hit(cell)
        self.boss_hp -= 1
        if self.boss_hp > 0:
            return 0.0
        self.enemies.remove(cell)
        self.boss_defeated = True
        return R_BOSS
    
    #added the trapdoor logic, where the trapdoor opens when the boss is defeated, and stepping on it completes the floor
    @property
    def trapdoor_open(self):
        return self.boss_defeated

#OBSERVATION AND INFO

    def door_label(self, d):
        #What lies behind door d of the current room.

        #WALL = 0: no door
        #EXPLORED = 1: no unvisited rooms in that branch behind that door
        #UNEXPLORED_BRANCH = 2: somewhere in that branch is a room that has not been visited
        #BOSS_NEXT_ROOM = 3: a door to the boss room
        
        nxt = self.doors[self.room].get(d)
        if nxt is None:
            return WALL
        if nxt == self.boss_room:
            return BOSS_NEXT_ROOM
        stack, seen = [nxt], {self.room, nxt}   # the floor is a tree: never go back
        while stack:
            room = stack.pop()
            if room not in self.visited:
                return UNEXPLORED_BRANCH
            for other in self.doors[room].values():
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        return EXPLORED

    def _get_obs(self):
        ar, ac = self.agent
        cell = ar * SIZE + ac
        if not self.enemies:    # navigation mode
            labels = [self.door_label(d) for d in range(4)]
            return self.combat_n + int(np.ravel_multi_index(
                (cell, *labels, int(self.trapdoor_open and self.room == self.boss_room)),
                self.nav_dims))
        er, ec = min(self.enemies, key=lambda e: abs(e[0] - ar) + abs(e[1] - ac))
        off_r = int(np.clip(er - ar, -2, 2)) + 2
        off_c = int(np.clip(ec - ac, -2, 2)) + 2
        left = self.boss_hp if self.room == self.boss_room else len(self.enemies)
        hearts = int(np.clip(self.hearts, 1, 3))    # 0 hearts (dead) reads as 1
        return int(np.ravel_multi_index(
            (cell, off_r * 5 + off_c, self.cooldown, min(left, 3) - 1, hearts - 1),
            self.combat_dims))

    def _info(self):
        return {"hearts": self.hearts, "enemies_left": len(self.enemies),
                "cleared": self.cleared, "room": self.room,
                "rooms_visited": len(self.visited), "boss_hp": self.boss_hp,
                "boss_defeated": self.boss_defeated, "floor_completed": self.floor_completed,
                "dead": self.hearts <= 0, "exited": False}

    #RENDERING
    def door_kinds(self):
        #Type of doors in each direction N,S,W,E: None (wall), 'door' or 'boss' 
        return [None if d not in self.doors[self.room] else
                "boss" if self.doors[self.room][d] == self.boss_room else "door"
                for d in range(4)]

    def map_cells(self):
        #Minimap: visited rooms, plus rooms seen through a door of a visited room.
        cells = {}
        for room in self.visited:
            for nxt in self.doors[room].values():
                cells.setdefault(nxt, "seen")
        for room in self.visited:
            cells[room] = "visited"
        cells[self.room] = "current"
        return cells

    def _render_text(self):
        text = super()._render_text().split("\n")
        grid = [list(row) for row in text[:7]]
        for d, kind in enumerate(self.door_kinds()):           # walls without a door
            if kind is None:
                r, c = [(0, 3), (6, 3), (3, 0), (3, 6)][d]
                grid[r][c] = "#"
        for r, c in self.rocks:
            grid[r + 1][c + 1] = "o"
        if self.trapdoor_open and self.room == self.boss_room and self.agent != TRAPDOOR:
            grid[TRAPDOOR[0] + 1][TRAPDOOR[1] + 1] = "T"
        status = f"room {self.room}  visited {len(self.visited)}/{len(self.rooms)}"
        if self.room == self.boss_room and not self.boss_defeated:
            status += f"  boss hp {self.boss_hp}"
        return "\n".join(["".join(row) for row in grid] + text[7:] + [status])
