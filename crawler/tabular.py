"""Tabular SARSA, Expected SARSA and Q-Learning (adapted from the course's deustorl code).

Differences from the course versions, all deliberate:
- true terminal transitions use target = r (no bootstrap from the next state);
- time-limit truncation DOES bootstrap: the timeout is not part of the game, and the
  observation has no clock, so the next state's value is the right estimate;
- greedy ties are broken uniformly at random with a seeded generator (not "first max");
- the Expected SARSA expectation counts every tied greedy action.
"""
import numpy as np

ALGOS = ("sarsa", "expected_sarsa", "q_learning")


def greedy(q_row, rng):
    """Index of a maximal entry, ties broken uniformly at random."""
    best = np.flatnonzero(q_row == q_row.max())
    return int(best[0]) if len(best) == 1 else int(rng.choice(best))


def epsilon_greedy(q_row, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(len(q_row)))
    return greedy(q_row, rng)


def expected_value(q_row, eps):
    """sum_a pi(a|s) Q(s,a) for the epsilon-greedy policy with random tie-breaking."""
    is_max = q_row == q_row.max()
    probs = np.full(len(q_row), eps / len(q_row))
    probs[is_max] += (1 - eps) / is_max.sum()
    return float(probs @ q_row)


def td_target(algo, q, reward, next_obs, next_action, eps, gamma, terminated):
    if terminated:
        return reward
    if algo == "sarsa":
        return reward + gamma * q[next_obs, next_action]
    if algo == "expected_sarsa":
        return reward + gamma * expected_value(q[next_obs], eps)
    if algo == "q_learning":
        return reward + gamma * q[next_obs].max()
    raise ValueError(f"unknown algorithm {algo!r}, expected one of {ALGOS}")


def linear_epsilon(t, steps, start=1.0, end=0.05, frac=0.7):
    """Linear decay from `start` at t=0 to `end` at t=frac*steps, then constant."""
    return start + (end - start) * min(1.0, t / (frac * steps))
