"""Play Basement Crawler in a window.

WASD move, arrow keys shoot (like The Binding of Isaac), Space waits,
R restarts, Esc quits. The game is turn-based: each key press is one step.
Usage: python play.py [Crawler-v1|Crawler-v2]   (default Crawler-v2)
"""
import sys

import pygame
import gymnasium as gym

import crawler  # noqa: F401  (registers the Crawler-v* ids)

KEYS = {
    pygame.K_w: 0, pygame.K_s: 1, pygame.K_a: 2, pygame.K_d: 3,              # move
    pygame.K_UP: 4, pygame.K_DOWN: 5, pygame.K_LEFT: 6, pygame.K_RIGHT: 7,   # shoot
    pygame.K_SPACE: 8,                                                       # wait
}

env = gym.make(sys.argv[1] if len(sys.argv) > 1 else "Crawler-v2", render_mode="human")
env.unwrapped.render_hint = True          # show "R: play again" after the episode
obs, info = env.reset()
total, done = 0.0, False

running = True
while running:
    event = pygame.event.wait()
    if event.type == pygame.QUIT:
        running = False
    elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
            running = False
        elif event.key == pygame.K_r:
            obs, info = env.reset()
            total, done = 0.0, False
        elif event.key in KEYS and not done:
            obs, reward, terminated, truncated, info = env.step(KEYS[event.key])
            total += reward
            done = terminated or truncated
            if done:
                result = ("exited" if info["exited"] else "floor complete" if info.get("floor_completed")
                          else "died" if info["hearts"] <= 0 else "time up")
                print(f"Episode over ({result}): return {total:+.2f}")
env.close()
