from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tensorboard.backend.event_processing import event_accumulator


ALGORITHMS = {
    "sarsa": "SARSA",
    "expected_sarsa": "Expected SARSA",
    "qlearning": "Q-Learning",
}

# Colores parecidos a los de TensorBoard
COLORS = {
    "sarsa": "#4FC3F7",           # azul claro
    "expected_sarsa": "#FF7043",  # naranja
    "qlearning": "#66BB6A",       # verde
}

METRICS = {
    "eval/success_rate": "Success rate",
    "eval/mean_return": "Mean return",
    "eval/death_rate": "Death rate",
    "eval/timeout_rate": "Timeout rate",
    "eval/mean_rooms_visited": "Mean rooms visited",
    "eval/mean_floors_completed": "Mean floors completed",
}

EXPERIMENTS = [
    "v1_main",
    "v2_main",
    "v3_main",
    "v3_10m",
]


def ema_smooth(values, weight=0.0):
    """
    TensorBoard-like smoothing.
    weight=0.0 -> no smoothing
    weight close to 1.0 -> more smoothing
    """
    if weight <= 0.0:
        return values.copy()

    smoothed = np.zeros_like(values, dtype=float)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = weight * smoothed[i - 1] + (1.0 - weight) * values[i]
    return smoothed


def read_metrics(run_directory):
    accumulator = event_accumulator.EventAccumulator(
        str(run_directory),
        size_guidance={"scalars": 0}
    )
    accumulator.Reload()

    tags = set(accumulator.Tags().get("scalars", []))
    result = {}

    for tag in METRICS:
        if tag not in tags:
            continue

        events = accumulator.Scalars(tag)
        if not events:
            continue

        steps = np.array([e.step for e in events], dtype=np.int64)
        values = np.array([e.value for e in events], dtype=float)
        result[tag] = (steps, values)

    return result


def aggregate_runs_for_metric(run_metric_list):
    """
    run_metric_list: list of (steps, values)
    Returns shared_steps, stacked_values with shape [num_runs, num_points]
    """
    if not run_metric_list:
        return None, None

    shared_steps = sorted(
        set.intersection(*(set(steps.tolist()) for steps, _ in run_metric_list))
    )

    if not shared_steps:
        return None, None

    shared_steps = np.array(shared_steps, dtype=np.int64)

    aligned_values = []
    for steps, values in run_metric_list:
        value_map = dict(zip(steps.tolist(), values.tolist()))
        aligned_values.append([value_map[int(s)] for s in shared_steps])

    return shared_steps, np.array(aligned_values, dtype=float)


def millions_formatter(x, pos):
    if x >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"{x/1_000:.0f}k"
    return f"{int(x)}"


def setup_tensorboard_like_style():
    plt.rcParams.update({
        "figure.facecolor": "#202124",
        "axes.facecolor": "#202124",
        "axes.edgecolor": "#BDBDBD",
        "axes.labelcolor": "#E0E0E0",
        "text.color": "#E0E0E0",
        "xtick.color": "#E0E0E0",
        "ytick.color": "#E0E0E0",
        "grid.color": "#5F6368",
        "grid.alpha": 0.35,
        "legend.facecolor": "#2B2B2B",
        "legend.edgecolor": "#888888",
        "savefig.facecolor": "#202124",
        "savefig.edgecolor": "#202124",
        "font.size": 11,
    })


def plot_metric(experiment, metric, all_data, output_dir, smoothing=0.6, show_seed_lines=True):
    setup_tensorboard_like_style()

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    mean_handles = []
    mean_labels = []

    right_text_lines = []
    plotted_any = False

    for algo_key, algo_label in ALGORITHMS.items():
        metric_runs = []

        for run_metrics in all_data.get(algo_key, []):
            if metric in run_metrics:
                metric_runs.append(run_metrics[metric])

        if not metric_runs:
            continue

        steps, values_stack = aggregate_runs_for_metric(metric_runs)
        if steps is None:
            continue

        color = COLORS[algo_key]

        # Dibujar cada seed en fino
        if show_seed_lines:
            for seed_idx, seed_values in enumerate(values_stack):
                seed_smooth = ema_smooth(seed_values, smoothing)
                ax.plot(
                    steps,
                    seed_smooth,
                    color=color,
                    linewidth=1.2,
                    alpha=0.28,
                )

        # Media y desviación
        mean_values = values_stack.mean(axis=0)
        std_values = values_stack.std(axis=0)

        mean_smooth = ema_smooth(mean_values, smoothing)
        upper_smooth = ema_smooth(mean_values + std_values, smoothing)
        lower_smooth = ema_smooth(mean_values - std_values, smoothing)

        handle, = ax.plot(
            steps,
            mean_smooth,
            color=color,
            linewidth=3.0,
        )

        ax.fill_between(
            steps,
            lower_smooth,
            upper_smooth,
            color=color,
            alpha=0.16,
        )

        mean_handles.append(handle)
        mean_labels.append(f"{algo_label} (mean)")
        plotted_any = True

        final_mean = mean_values[-1]
        final_std = std_values[-1]
        right_text_lines.append(f"{algo_label}: {final_mean:.3f} ± {final_std:.3f}")

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_title(f"{experiment}: {METRICS[metric]}", fontsize=14, pad=12)
    ax.set_xlabel("Training steps")
    ax.set_ylabel(METRICS[metric])

    ax.grid(True)
    ax.xaxis.set_major_formatter(FuncFormatter(millions_formatter))

    if metric.endswith("_rate"):
        ax.set_ylim(0, 1.02)

    # Leyenda principal
    legend1 = ax.legend(
        mean_handles,
        mean_labels,
        loc="lower left",
        fontsize=10,
        framealpha=0.9
    )
    ax.add_artist(legend1)


    fig.tight_layout()

    safe_metric = metric.replace("eval/", "")
    png_path = output_dir / f"{experiment}_{safe_metric}.png"
    pdf_path = output_dir / f"{experiment}_{safe_metric}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    plt.close(fig)
    print(f"Created: {png_path}")
    print(f"Created: {pdf_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("figures/rl_better"))
    parser.add_argument("--experiments", nargs="+", default=EXPERIMENTS)
    parser.add_argument("--smoothing", type=float, default=0.6)
    parser.add_argument("--hide-seeds", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    for experiment in args.experiments:
        exp_dir = args.runs_dir / experiment

        if not exp_dir.is_dir():
            print(f"Skipping missing experiment: {exp_dir}")
            continue

        all_data = {}

        for algo_key in ALGORITHMS:
            algo_dir = exp_dir / algo_key
            seed_dirs = sorted(algo_dir.glob("seed*")) if algo_dir.exists() else []

            all_data[algo_key] = []
            for seed_dir in seed_dirs:
                if seed_dir.is_dir():
                    all_data[algo_key].append(read_metrics(seed_dir))

        for metric in METRICS:
            plot_metric(
                experiment=experiment,
                metric=metric,
                all_data=all_data,
                output_dir=args.out,
                smoothing=args.smoothing,
                show_seed_lines=not args.hide_seeds,
            )

    print("\nFinished exporting figures.")


if __name__ == "__main__":
    main()