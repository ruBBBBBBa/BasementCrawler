"""Basement Crawler v1: one room, chasing enemies, travelling tears.

Grid: 5x5 room, a door in the centre of each wall (locked until the room is clear).
Actions: 0-3 move N,S,W,E | 4-7 shoot N,S,W,E | 8 wait.
Observation: one integer (spaces.Discrete), see _get_obs().
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

SIZE = 5
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]                  # N, S, W, E
DOOR_CELLS = {0: (0, 2), 1: (4, 2), 2: (2, 0), 3: (2, 4)}   # cell in front of each door
START = (4, 2)                                             # agent enters from the south door

R_KILL, R_CLEAR, R_EXIT = 1.0, 2.0, 5.0
R_HIT, R_DEATH, R_STEP = -3.0, -10.0, -0.01


class CrawlerV1Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 6}

    def __init__(self, n_enemies=3, enemy_move_prob=0.5, tear_speed=2,
                 tear_range=4, fire_delay=2, hearts=3, render_mode=None):
        assert 1 <= n_enemies <= 3, "v1 observation supports 1-3 enemies"
        self.n_enemies, self.enemy_move_prob = n_enemies, enemy_move_prob
        self.tear_speed, self.tear_range, self.fire_delay = tear_speed, tear_range, fire_delay
        self.max_hearts, self.render_mode = hearts, render_mode
        # combat: agent cell x nearest-enemy offset x cooldown x enemies left
        self.combat_dims = (SIZE * SIZE, 25, fire_delay + 1, 3)
        self.combat_n = int(np.prod(self.combat_dims))
        # + one navigation state per agent cell once the room is clear
        self.observation_space = spaces.Discrete(self.combat_n + SIZE * SIZE)
        self.action_space = spaces.Discrete(9)
        self._renderer = None                 # pygame graphics, created on first render
        self.render_hint = False              # play.py sets this to show "R: play again"

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)              # seeds self.np_random
        self.agent, self.hearts, self.cooldown = START, self.max_hearts, 0
        self.tear = None                      # [cell, direction, cells left] or None
        self.cleared = self.exited = False
        far = [(r, c) for r in range(SIZE) for c in range(SIZE)
               if abs(r - START[0]) + abs(c - START[1]) >= 3]
        idx = self.np_random.choice(len(far), size=self.n_enemies, replace=False)
        self.enemies = [far[i] for i in idx]  # (row, col) of living enemies
        self._render_human()
        return self._get_obs(), self._info()

    def step(self, action):
        action, reward = int(action), R_STEP
        if self.cooldown > 0:                                   # 1. cooldown ticks
            self.cooldown -= 1
        if action < 4:                                          # 2. agent acts
            reward += self._move_agent(action)
        elif action < 8 and self.cooldown == 0 and self.tear is None:
            self.tear = [self.agent, action - 4, self.tear_range]
            self.cooldown = self.fire_delay
        if self.exited:
            self._render_human()
            return self._get_obs(), reward + R_EXIT, True, False, self._info()
        reward += self._advance_tear()                          # 3. tear flies
        reward += self._move_enemies()                          # 4. enemies chase
        if not self.enemies and not self.cleared:               # 5. checks
            self.cleared = True
            reward += R_CLEAR
        terminated = self.hearts <= 0
        if terminated:
            reward += R_DEATH
        self._render_human()
        return self._get_obs(), reward, terminated, False, self._info()

    def _move_agent(self, d):
        r, c = self.agent
        nr, nc = r + DIRS[d][0], c + DIRS[d][1]
        if not (0 <= nr < SIZE and 0 <= nc < SIZE):
            if self.cleared and self.agent == DOOR_CELLS[d]:
                return self._use_door(d)              # walked through an open door
            return 0.0                                # otherwise bumped a wall
        if self._blocked((nr, nc)):
            return 0.0
        if (nr, nc) in self.enemies:
            self.hearts -= 1                          # touching an enemy hurts
            return R_HIT
        self.agent = (nr, nc)
        return 0.0

    def _use_door(self, d):
        self.exited = True
        return 0.0

    def _blocked(self, cell):
        """Cells inside the room that stop movement and tears (rocks from v2)."""
        return False

    def _hit(self, cell):
        """A tear struck the enemy at `cell`."""
        self.enemies.remove(cell)
        return R_KILL

    def _advance_tear(self):
        if self.tear is None:
            return 0.0
        (r, c), d, left = self.tear
        for _ in range(self.tear_speed):              # one cell at a time: no tunnelling
            r, c = r + DIRS[d][0], c + DIRS[d][1]
            left -= 1
            if (not (0 <= r < SIZE and 0 <= c < SIZE) or self._blocked((r, c))
                    or (left == 0 and (r, c) not in self.enemies)):
                self.tear = None
                return 0.0
            if (r, c) in self.enemies:
                self.tear = None
                return self._hit((r, c))
        self.tear = [(r, c), d, left]
        return 0.0

    def _move_enemies(self):
        reward = 0.0
        for enemy in list(self.enemies):
            if self.hearts <= 0:
                break
            if self.np_random.random() >= self.enemy_move_prob:
                continue
            dr, dc = self.agent[0] - enemy[0], self.agent[1] - enemy[1]
            if abs(dr) > abs(dc) or (abs(dr) == abs(dc) and self.np_random.random() < 0.5):
                target = (enemy[0] + int(np.sign(dr)), enemy[1])
            else:
                target = (enemy[0], enemy[1] + int(np.sign(dc)))
            if target == self.agent:
                self.hearts -= 1                      # bump: enemy stays put
                reward += R_HIT
            elif target not in self.enemies and not self._blocked(target):  # no sharing cells
                self.enemies[self.enemies.index(enemy)] = target
                if self.tear is not None and self.tear[0] == target:
                    self.tear = None                  # walked into the tear
                    reward += self._hit(target)
        return reward

    def _get_obs(self):
        ar, ac = self.agent
        cell = ar * SIZE + ac
        if not self.enemies:                          # navigation mode
            return self.combat_n + cell
        er, ec = min(self.enemies, key=lambda e: abs(e[0] - ar) + abs(e[1] - ac))
        off_r = int(np.clip(er - ar, -2, 2)) + 2
        off_c = int(np.clip(ec - ac, -2, 2)) + 2
        left = min(len(self.enemies), 3) - 1
        return int(np.ravel_multi_index((cell, off_r * 5 + off_c, self.cooldown, left),
                                        self.combat_dims))

    def _info(self):
        return {"hearts": self.hearts, "enemies_left": len(self.enemies),
                "cleared": self.cleared, "exited": self.exited}

    # ------------------------------------------------------------ rendering
    def render(self):
        if self.render_mode == "ansi":
            return self._render_text()
        if self.render_mode in ("human", "rgb_array"):
            if self._renderer is None:
                from .render import Renderer          # pygame is only needed for graphics
                self._renderer = Renderer(self.render_mode, self.metadata["render_fps"])
            return self._renderer.draw(self)

    def _render_human(self):
        if self.render_mode == "human":
            self.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _render_text(self):
        door = "." if self.cleared else "+"
        rows = ["#" * 3 + door + "#" * 3]
        for r in range(SIZE):
            line = door if r == 2 else "#"
            for c in range(SIZE):
                if (r, c) == self.agent:
                    line += "A"
                elif (r, c) in self.enemies:
                    line += "E"
                elif self.tear is not None and self.tear[0] == (r, c):
                    line += "*"
                else:
                    line += "."
            rows.append(line + (door if r == 2 else "#"))
        rows.append("#" * 3 + door + "#" * 3)
        rows.append(f"hearts {self.hearts}  enemies {len(self.enemies)}  cooldown {self.cooldown}")
        return "\n".join(rows)
