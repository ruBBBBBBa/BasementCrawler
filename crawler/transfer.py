"""Map a Crawler-v1 Q-table onto the Crawler-v2 observation encoding.

v1 combat index = ravel(cell, offset, cooldown, left)            dims (25, 25, 3, 3)
v2 combat index = ravel(cell, offset, cooldown, left, hearts)     dims (25, 25, 3, 3, 3)
The first four features are computed the same way in both versions (tests check this on
matched game states), so every v2 combat row copies the v1 row with the same four
features, for each of the three heart values. v1 never sees hearts, so its values are
averaged over whatever heart levels it met in training.

Caveat: in the v2 boss room `left` is the boss's HP, which v1 never had; a v1 row for
"k chasers left" is copied to "boss has k HP" because the agent cannot tell them apart.

Navigation rows are not compatible (v1: cell only; v2: cell + door labels + trapdoor)
and keep the normal initial value.
"""
import numpy as np


def v1_to_v2_q(q1, env1, env2, nav_init=0.0):
    """Return a v2 Q-table built from v1 table `q1`; env1/env2 are unwrapped envs."""
    dims1, dims2 = tuple(env1.combat_dims), tuple(env2.combat_dims)
    if dims2[:4] != dims1 or q1.shape != (env1.observation_space.n, env1.action_space.n):
        raise ValueError(f"incompatible encodings: v1 {dims1} {q1.shape}, v2 {dims2}")
    q2 = np.full((env2.observation_space.n, env2.action_space.n), nav_init, dtype=q1.dtype)
    features = np.unravel_index(np.arange(env2.combat_n), dims2)
    q2[:env2.combat_n] = q1[np.ravel_multi_index(features[:4], dims1)]
    return q2
