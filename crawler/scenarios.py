"""Named Crawler-v2 floors for curriculum learning.

Each scenario is a sub-tree of the real v2 floor ending at the boss room, passed to
CrawlerV2Env's existing constructor arguments. Rooms, rocks, enemy counts, rewards and
the observation/action encoding are unchanged; only the set of rooms and the start room
differ. Doors that lead to rooms outside the scenario become walls (door label WALL).

    full               start -> medium -> hard -> boss  (+ easy dead end)
    medium_hard_boss          medium -> hard -> boss
    hard_boss                           hard -> boss
    boss_only                                   boss
"""
from .crawler_v2 import ROOMS, EDGES, BOSS_ROOM

MEDIUM, HARD = (1, 2), (1, 3)


def _sub_floor(start, rooms):
    return {"rooms": {r: ROOMS[r] for r in rooms},
            "edges": [e for e in EDGES if e[0] in rooms and e[1] in rooms],
            "start_room": start, "boss_room": BOSS_ROOM}


SCENARIOS = {
    "full": {},
    "medium_hard_boss": _sub_floor(MEDIUM, (MEDIUM, HARD, BOSS_ROOM)),
    "hard_boss": _sub_floor(HARD, (HARD, BOSS_ROOM)),
    "boss_only": _sub_floor(BOSS_ROOM, (BOSS_ROOM,)),
}


def make_kwargs(env_id, scenario):
    """gym.make keyword arguments for a scenario name (only Crawler-v2 has scenarios)."""
    if scenario in (None, "full"):
        return {}
    if env_id != "Crawler-v2":
        raise ValueError(f"scenario {scenario!r} only exists for Crawler-v2")
    return dict(SCENARIOS[scenario])
