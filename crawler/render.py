"""Pygame graphics for Basement Crawler.

Used by render_mode="human" (a window) and render_mode="rgb_array" (frames for video).
"""
import sys

import numpy as np
import pygame

#CANVAS
TILE = 64
HUD = 56
COLS = ROWS = 7 # 5x5 room + a 1-tile wall all round
WIDTH, HEIGHT = COLS * TILE, HUD + ROWS * TILE  # 448 x 504 (even sizes, safe for video)
DOORS = [(0, 3), (6, 3), (3, 0), (3, 6)]    # wall tiles holding the N, S, W, E doors
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # N, S, W, E

#COLORS
BG = (18, 14, 12)
WALL, WALL_LINE = (62, 50, 44), (44, 35, 30)
FLOOR_A, FLOOR_B, FLOOR_LINE = (122, 94, 68), (112, 86, 62), (98, 75, 54)
DOOR_WOOD, DOOR_BAR, DOOR_OPEN, DOOR_FRAME = (104, 62, 32), (64, 36, 18), (6, 5, 4), (150, 118, 76)
HERO, HERO_EDGE = (70,190,110), (34, 104, 124)
ENEMY, ENEMY_EDGE = (176, 44, 54), (104, 20, 30)
TEAR, TEAR_EDGE = (160, 214, 255), (74, 144, 214)
HEART, HEART_EMPTY = (224, 42, 64), (72, 34, 38)
TEXT, TEXT_DIM = (236, 226, 212), (160, 150, 140)
WHITE, PUPIL, HIT = (246, 246, 246), (20, 14, 14), (255, 70, 70)
ROCK, ROCK_EDGE = (128, 120, 112), (78, 72, 66)
BOSS, BOSS_EDGE, BOSS_DOOR = (120, 40, 140), (70, 18, 84), (170, 30, 40)
MAP_ROOM = {"current": (236, 226, 212), "visited": (130, 120, 110), "seen": (60, 52, 46)}

#CONVERSION FROM GRID CELLS TO SCREEN PIXELS
def _center(cell):
    #Screen centre of an interior cell (row, col).
    return (cell[1] + 1) * TILE + TILE // 2, HUD + (cell[0] + 1) * TILE + TILE // 2


class Renderer:
    def __init__(self, mode, fps):
        self.mode, self.fps = mode, fps
        pygame.font.init() # initialize the text rendering module
        if mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Basement Crawler")
            self.clock = pygame.time.Clock()
        self.surface = pygame.Surface((WIDTH, HEIGHT))
        self.font = pygame.font.Font(None, 28)
        self.small = pygame.font.Font(None, 22)
        self.big = pygame.font.Font(None, 48)
        self.tiny = pygame.font.Font(None, 18)
        self.prev = None    # (agent, n_enemies, hearts) from the previous frame
        self.facing = 0    # direction the hero looks (purely visual)

    #MAIN 
    def draw(self, env):
        hit = self._track(env)
        s = self.surface
        s.fill(BG)
        kinds = env.door_kinds() if hasattr(env, "door_kinds") else ["door"] * 4
        self._room(s, env.cleared, kinds)
        for rock in getattr(env, "rocks", ()):
            self._rock(s, rock)
        if getattr(env, "trapdoor_open", False) and env.room == env.boss_room:
            self._trapdoor(s)
        boss = getattr(env, "room", None) == getattr(env, "boss_room", ()) and env.boss_hp > 0
        for enemy in env.enemies:
            self._enemy(s, enemy, env.agent, boss)
        self._hero(s, env.agent, hit)
        if env.tear is not None:
            self._tear(s, env.tear[0], env.tear[1])
        self._hud(s, env)
        hint = getattr(env, "render_hint", False)
        if hasattr(env, "map_cells"):
            self._minimap(s, env.map_cells(), env.boss_room)
        if getattr(env, "n_floors", 1) > 1:    # v3: which floor we are on, under the minimap
            label = self.tiny.render(f"Floor {env.floor + 1}/{env.n_floors}", True, TEXT)
            s.blit(label, (4, HUD + 62))
        if env.hearts <= 0:
            self._banner(s, "You died", HIT, hint)
        elif getattr(env, "run_completed", False):
            self._banner(s, "You escaped!", TEAR, hint)
        elif getattr(env, "floor_completed", False):
            self._banner(s, "Floor complete!", TEAR, hint)
        elif env.exited:
            self._banner(s, "Room cleared!", TEAR, hint)
        return self._output()

    def close(self):
        if self.mode == "human":
            pygame.display.quit()

    #HELPERS
    def _track(self, env):
        #Work out visual-only details by comparing frames.
        new_episode = (self.prev is None or env.hearts > self.prev[2]
                       or len(env.enemies) > self.prev[1])
        hit = not new_episode and env.hearts < self.prev[2]
        if env.tear is not None:
            self.facing = env.tear[1]
        elif not new_episode and env.agent != self.prev[0]:
            dr, dc = env.agent[0] - self.prev[0][0], env.agent[1] - self.prev[0][1]
            self.facing = DIRS.index((dr, dc)) if (dr, dc) in DIRS else self.facing
        elif new_episode:
            self.facing = 0
        self.prev = (env.agent, len(env.enemies), env.hearts)
        return hit

    def _output(self):
        if self.mode == "human":
            if pygame.event.get(pygame.QUIT):   # closing the window ends the script
                pygame.display.quit()
                sys.exit(0)
            pygame.event.pump()
            self.window.blit(self.surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.fps)
            return None
        return np.transpose(np.array(pygame.surfarray.pixels3d(self.surface)), (1, 0, 2))

    def _room(self, s, cleared, kinds):
        pygame.draw.rect(s, WALL, (0, HUD, WIDTH, ROWS * TILE))
        for y in range(HUD, HEIGHT, TILE // 2): # brick courses
            pygame.draw.line(s, WALL_LINE, (0, y), (WIDTH, y), 2)
        for r in range(5):  # checkered floor
            for c in range(5):
                x, y = (c + 1) * TILE, HUD + (r + 1) * TILE
                pygame.draw.rect(s, FLOOR_A if (r + c) % 2 == 0 else FLOOR_B, (x, y, TILE, TILE))
                pygame.draw.rect(s, FLOOR_LINE, (x, y, TILE, TILE), 1)
        pygame.draw.rect(s, WALL_LINE, (TILE - 3, HUD + TILE - 3, 5 * TILE + 6, 5 * TILE + 6), 3)
        for (r, c), kind in zip(DOORS, kinds):
            if kind is None:    # no door this side
                continue
            x, y = c * TILE + 8, HUD + r * TILE + 8
            rect = pygame.Rect(x, y, TILE - 16, TILE - 16)
            frame = BOSS_DOOR if kind == "boss" else DOOR_FRAME
            pygame.draw.rect(s, frame, rect.inflate(8, 8), border_radius=6)
            if cleared:
                pygame.draw.rect(s, DOOR_OPEN, rect, border_radius=4)
            else:
                pygame.draw.rect(s, DOOR_WOOD, rect, border_radius=4)
                for i in range(1, 3):   # iron bars
                    if r in (0, 6):
                        bx = x + i * rect.w // 3
                        pygame.draw.line(s, DOOR_BAR, (bx, y), (bx, y + rect.h), 4)
                    else:
                        by = y + i * rect.h // 3
                        pygame.draw.line(s, DOOR_BAR, (x, by), (x + rect.w, by), 4)

    @staticmethod
    def _eyes(s, cx, cy, look, spread=9, size=7):
        lx, ly = look
        n = max((lx * lx + ly * ly) ** 0.5, 1e-6)
        px, py = round(3 * lx / n), round(3 * ly / n)
        for ex in (cx - spread, cx + spread):
            pygame.draw.circle(s, WHITE, (ex, cy - 4), size)
            pygame.draw.circle(s, PUPIL, (ex + px, cy - 4 + py), size // 2 + 1)

    def _hero(self, s, cell, hit):
        cx, cy = _center(cell)
        pygame.draw.ellipse(s, (0, 0, 0), (cx - 20, cy + 16, 40, 10))   # shadow
        pygame.draw.circle(s, HERO_EDGE, (cx, cy), 25)
        pygame.draw.circle(s, HERO, (cx, cy), 22)
        pygame.draw.circle(s, (150, 222, 236), (cx - 9, cy - 11), 5)    # shine
        dr, dc = DIRS[self.facing]
        self._eyes(s, cx, cy, (dc, dr))
        if hit:
            pygame.draw.circle(s, HIT, (cx, cy), 29, 4)

    def _enemy(self, s, cell, agent, boss=False):
        cx, cy = _center(cell)
        ax, ay = _center(agent)
        pygame.draw.ellipse(s, (0, 0, 0), (cx - 18, cy + 16, 36, 9))    # shadow
        if boss:    # bigger, horned
            for side in (-1, 1):
                pygame.draw.polygon(s, BOSS_EDGE, [(cx + side * 12, cy - 18),
                                                   (cx + side * 26, cy - 30), (cx + side * 22, cy - 10)])
            pygame.draw.circle(s, BOSS_EDGE, (cx, cy), 28)
            pygame.draw.circle(s, BOSS, (cx, cy), 25)
        else:
            pygame.draw.circle(s, ENEMY_EDGE, (cx, cy), 23)
            pygame.draw.circle(s, ENEMY, (cx, cy), 20)
        for x in range(cx - 14, cx + 15, 7):    # jagged teeth
            pygame.draw.polygon(s, WHITE, [(x, cy + 6), (x + 6, cy + 6), (x + 3, cy + 12)])
        self._eyes(s, cx, cy - 2, (ax - cx, ay - cy), spread=8, size=6)
        pygame.draw.line(s, PUPIL, (cx - 15, cy - 15), (cx - 3, cy - 10), 3)  # angry brows
        pygame.draw.line(s, PUPIL, (cx + 15, cy - 15), (cx + 3, cy - 10), 3)

    @staticmethod
    def _rock(s, cell):
        cx, cy = _center(cell)
        pygame.draw.circle(s, ROCK_EDGE, (cx, cy + 2), 26)
        pygame.draw.circle(s, ROCK, (cx - 2, cy - 1), 22)
        pygame.draw.circle(s, (160, 152, 144), (cx - 9, cy - 9), 6)

    @staticmethod
    def _trapdoor(s):
        cx, cy = _center((2, 2))
        pygame.draw.rect(s, DOOR_FRAME, (cx - 26, cy - 26, 52, 52), border_radius=6)
        pygame.draw.rect(s, DOOR_OPEN, (cx - 21, cy - 21, 42, 42), border_radius=4)

    @staticmethod
    def _minimap(s, cells, boss_room, cell=11):
        #Floor map in the top-left wall corner (5x5 floor grid).
        for (r, c), status in cells.items():
            rect = (4 + c * cell, HUD + 4 + r * cell, cell - 2, cell - 2)
            pygame.draw.rect(s, MAP_ROOM[status], rect)
            if (r, c) == boss_room and status != "current":
                pygame.draw.rect(s, BOSS_DOOR, rect, 2)

    @staticmethod
    def _tear(s, cell, direction):
        cx, cy = _center(cell)
        dr, dc = DIRS[direction]
        for k, rad in ((2, 4), (1, 6)):    # trail
            pygame.draw.circle(s, TEAR_EDGE, (cx - dc * 12 * k, cy - dr * 12 * k), rad)
        pygame.draw.circle(s, TEAR_EDGE, (cx, cy), 11)
        pygame.draw.circle(s, TEAR, (cx, cy), 9)
        pygame.draw.circle(s, WHITE, (cx - 3, cy - 3), 3)

    @staticmethod
    def _heart(s, x, y, color):
        pygame.draw.circle(s, color, (x - 7, y), 8)
        pygame.draw.circle(s, color, (x + 7, y), 8)
        pygame.draw.polygon(s, color, [(x - 15, y + 2), (x + 15, y + 2), (x, y + 18)])

    def _hud(self, s, env):
        pygame.draw.rect(s, BG, (0, 0, WIDTH, HUD))
        for i in range(env.max_hearts):
            self._heart(s, 26 + i * 36, 22, HEART if i < env.hearts else HEART_EMPTY)
        if getattr(env, "room", None) == getattr(env, "boss_room", ()) and env.boss_hp > 0:
            label = self.font.render(f"Boss HP: {env.boss_hp}", True, HIT)
        elif env.cleared:
            label = self.font.render("Doors open", True, TEAR)
        else:
            label = self.font.render(f"Enemies: {len(env.enemies)}", True, TEXT)
        s.blit(label, label.get_rect(center=(WIDTH // 2 + 20, HUD // 2)))
        tag = self.small.render("Tear", True, TEXT_DIM)
        s.blit(tag, (WIDTH - 112, HUD // 2 - 7))
        for i in range(env.fire_delay):    #filled pips = shot ready
            ready = env.cooldown <= i
            pygame.draw.circle(s, TEAR if ready else (50, 60, 70), (WIDTH - 62 + i * 22, HUD // 2), 7)

    def _banner(self, s, text, color, hint):
        shade = pygame.Surface((WIDTH, 110), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 190))
        s.blit(shade, (0, HUD + 3 * TILE - 23))
        msg = self.big.render(text, True, color)
        s.blit(msg, msg.get_rect(center=(WIDTH // 2, HUD + 3 * TILE + 20)))
        if hint:
            hint = self.small.render("R: play again    Esc: quit", True, TEXT_DIM)
            s.blit(hint, hint.get_rect(center=(WIDTH // 2, HUD + 3 * TILE + 58)))
