
"""Watch or record a trained Basement Crawler agent."""

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import crawler


def load_agent(qtable_path, random_agent, random_env):
    if random_agent:
        return None, random_env, {}

    if qtable_path is None:
        raise ValueError("Provide a Q-table, or use --random.")

    config_path = qtable_path.with_suffix(".json")

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing {config_path}. Keep the JSON file next "
            "to the Q-table to load the correct environment."
        )

    config = json.loads(config_path.read_text())

    return (
        np.load(qtable_path, allow_pickle=False),
        config["env_id"],
        config["env"],
    )


def choose_action(qtable, state, rng, n_actions):
    if qtable is None:
        return int(rng.integers(n_actions))

    row = qtable[int(state)]
    best = np.flatnonzero(row == row.max())

    return int(rng.choice(best))


def success(info, env_id):
    if env_id == "Crawler-v1":
        return bool(info.get("exited", False))

    if env_id == "Crawler-v2":
        return bool(info.get("floor_completed", False))

    # V3 requires completing BOTH floors.
    return bool(info.get("run_completed", False))


def play_episode(env, qtable, env_id, seed, policy_seed):
    rng = np.random.default_rng(policy_seed)
    state, _ = env.reset(seed=seed)

    total = 0.0
    steps = 0

    while True:
        action = choose_action(
            qtable, state, rng, env.action_space.n
        )

        state, reward, terminated, truncated, info = env.step(action)

        total += float(reward)
        steps += 1

        if terminated or truncated:
            break

    won = success(info, env_id)

    if won:
        outcome = "SUCCESS"
    elif info.get("hearts", 1) <= 0:
        outcome = "DEATH"
    else:
        outcome = "TIMEOUT"

    return {
        "seed": seed,
        "policy_seed": policy_seed,
        "success": won,
        "outcome": outcome,
        "steps": steps,
        "return": round(total, 3),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "qtable", nargs="?", type=Path
    )
    parser.add_argument(
        "--random", action="store_true"
    )
    parser.add_argument(
        "--env", default="Crawler-v3"
    )
    parser.add_argument(
        "--record", type=Path, metavar="FOLDER"
    )
    parser.add_argument(
        "--episodes", type=int, default=1
    )
    parser.add_argument(
        "--seed", type=int
    )
    parser.add_argument(
        "--seed-start", type=int, default=10_000_000
    )
    parser.add_argument(
        "--policy-seed", type=int, default=0
    )
    parser.add_argument(
        "--find-win", action="store_true"
    )
    parser.add_argument(
        "--search-limit", type=int, default=100
    )
    parser.add_argument(
        "--pick",
        choices=("first", "shortest"),
        default="first",
    )
    parser.add_argument(
        "--fps", type=int, default=12
    )

    args = parser.parse_args()

    if args.qtable is None and not args.random:
        parser.error("Provide a Q-table or use --random")

    if args.random and args.qtable is not None:
        parser.error("Use either a Q-table or --random")

    if args.episodes < 1 or args.search_limit < 1 or args.fps < 1:
        parser.error("Episodes, search limit and FPS must be positive")

    qtable, env_id, env_kwargs = load_agent(
        args.qtable, args.random, args.env
    )

    first_seed = (
        args.seed if args.seed is not None
        else args.seed_start
    )

    # Validate Q-table dimensions.
    check_env = gym.make(env_id, **env_kwargs)

    try:
        if qtable is not None:
            expected_shape = (
                check_env.observation_space.n,
                check_env.action_space.n,
            )

            if qtable.shape != expected_shape:
                raise ValueError(
                    f"Q-table shape {qtable.shape} "
                    f"does not match {env_id}: {expected_shape}"
                )
    finally:
        check_env.close()

    # Search successful episodes WITHOUT rendering.
    if args.find_win:
        probe = gym.make(env_id, **env_kwargs)
        selected = None

        try:
            for i in range(args.search_limit):
                episode_seed = first_seed + i
                policy_seed = args.policy_seed + i

                result = play_episode(
                    probe,
                    qtable,
                    env_id,
                    episode_seed,
                    policy_seed,
                )

                if result["success"]:
                    if (
                        selected is None
                        or result["steps"] < selected["steps"]
                    ):
                        selected = result

                    if args.pick == "first":
                        break
        finally:
            probe.close()

        if selected is None:
            print(
                f"No wins among {args.search_limit} episodes. "
                "No video recorded."
            )
            return

        print(
            f"Selected: seed {selected['seed']}, "
            f"{selected['steps']} steps, "
            f"return {selected['return']:+.2f}"
        )

        to_play = [
            (selected["seed"], selected["policy_seed"])
        ]

    else:
        to_play = [
            (first_seed + i, args.policy_seed + i)
            for i in range(args.episodes)
        ]

    # Render ONLY selected episodes.
    render_mode = "rgb_array" if args.record else "human"

    env = gym.make(
        env_id,
        render_mode=render_mode,
        **env_kwargs,
    )

    if args.record:
        args.record.mkdir(parents=True, exist_ok=True)

        env.metadata = {
            **env.metadata,
            "render_fps": args.fps,
        }

        prefix = (
            args.qtable.stem if args.qtable
            else "random"
        )

        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(args.record),
            episode_trigger=lambda _: True,
            name_prefix=prefix,
        )

    results = []

    try:
        for seed, policy_seed in to_play:
            result = play_episode(
                env, qtable, env_id, seed, policy_seed
            )

            results.append(result)

            print(
                f"Seed {seed}: {result['outcome']}, "
                f"{result['steps']} steps, "
                f"return {result['return']:+.2f}"
            )

            if not args.record:
                time.sleep(1.5)

    finally:
        env.close()

    if args.record:
        summary = {
            "environment": env_id,
            "qtable": (
                str(args.qtable) if args.qtable
                else None
            ),
            "fps": args.fps,
            "episodes": results,
        }

        summary_path = args.record / "recording_results.json"

        summary_path.write_text(
            json.dumps(summary, indent=2)
        )

        print(f"Videos saved in {args.record}/")


if __name__ == "__main__":
    main()
