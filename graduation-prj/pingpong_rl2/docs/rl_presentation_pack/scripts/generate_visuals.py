from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "docs" / "rl_presentation_pack"
ASSET_DIR = OUT_DIR / "assets"
DATA_DIR = OUT_DIR / "data"


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def load_csv(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def savefig(name: str) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(ASSET_DIR / name, dpi=180)
    plt.close()


def metric_rows() -> list[dict[str, object]]:
    sources = [
        ("v25", "artifacts/ppo_runs/_legacy_models/pmk_cf_self_rally_v25/pmk_cf_self_rally_v25_training_summary.json"),
        ("v26", "artifacts/ppo_runs/keep1_v26/keep1_v26_training_summary.json"),
        ("v30", "artifacts/ppo_runs/keep1_v30/keep1_v30_training_summary.json"),
        ("v31", "artifacts/ppo_runs/keep1_v31/keep1_v31_training_summary.json"),
        ("v32 17D", "artifacts/ppo_runs/keep1_v32_17d/keep1_v32_17d_training_summary.json"),
        ("v34 17D", "artifacts/ppo_runs/keep1_v34_17d_long_xyz012/keep1_v34_17d_long_xyz012_training_summary.json"),
        ("v35 17D", "artifacts/ppo_runs/keep1_v35_17d_strong_axis_stable/keep1_v35_17d_strong_axis_stable_training_summary.json"),
    ]
    rows: list[dict[str, object]] = []
    for label, path in sources:
        data = load_json(path)
        config = data.get("config", {})
        eval_data = data.get("evaluation", {})
        action_mode = str(config.get("action_mode", ""))
        action_dim = 17 if "tracking" in action_mode else 15
        rows.append(
            {
                "model": label,
                "action_dim": action_dim,
                "reset_xy_range": config.get("reset_xy_range"),
                "reset_height": config.get("reset_ball_height_bounds"),
                "reset_velocity_xy_range": config.get("reset_velocity_xy_range"),
                "reset_velocity_z_range": config.get("reset_velocity_z_range"),
                "evaluation_step_limit": config.get("evaluation_step_limit") or config.get("max_episode_steps"),
                "stable_cycle_reward_cap": config.get("stable_cycle_reward_cap"),
                "low_apex_grace": config.get("low_apex_contact_grace_count"),
                "episodes": eval_data.get("episodes"),
                "mean_useful": eval_data.get("mean_useful_bounces"),
                "max_useful": eval_data.get("max_useful_bounces"),
                "rate_30_plus": eval_data.get("thirty_or_more_useful_bounce_rate"),
                "time_limit": eval_data.get("failure_counts", {}).get("time_limit", 0),
                "low_apex": eval_data.get("failure_counts", {}).get("low_apex_contact", 0),
                "ball_out": eval_data.get("failure_counts", {}).get("ball_out_of_bounds", 0),
                "robot_body": eval_data.get("failure_counts", {}).get("robot_body_contact", 0),
                "floor": eval_data.get("failure_counts", {}).get("floor_contact", 0),
                "ball_speed": eval_data.get("failure_counts", {}).get("ball_speed_limit", 0),
            }
        )
    return rows


def long_eval_rows() -> list[dict[str, object]]:
    rows = []
    for label, path in [
        ("v32 17D", "artifacts/ppo_runs/keep1_v32_17d/analysis/keep1_v32_17d_long7200_eval20_episodes.csv"),
        ("v33 17D", "artifacts/ppo_runs/keep1_v33_17d_perf/analysis/keep1_v33_17d_perf_long7200_eval20_episodes.csv"),
        ("v34 17D", "artifacts/ppo_runs/keep1_v34_17d_long_xyz012/analysis/keep1_v34_17d_long_xyz012_long7200_eval20_episodes.csv"),
        ("v35 17D", "artifacts/ppo_runs/keep1_v35_17d_strong_axis_stable/analysis/keep1_v35_17d_strong_axis_stable_long7200_eval20_episodes.csv"),
    ]:
        episodes = load_csv(path)
        contacts = [int(row["contact_count"]) for row in episodes]
        useful = [int(row["useful_bounces"]) for row in episodes]
        failures = Counter(row["failure_reason"] for row in episodes)
        rows.append(
            {
                "model": label,
                "episodes": len(episodes),
                "mean_contacts": float(np.mean(contacts)),
                "mean_useful": float(np.mean(useful)),
                "max_contacts": max(contacts),
                "max_useful": max(useful),
                "contacts300_useful100": sum(c >= 300 and u >= 100 for c, u in zip(contacts, useful)),
                "contacts400_useful150": sum(c >= 400 and u >= 150 for c, u in zip(contacts, useful)),
                "time_limit": failures.get("time_limit", 0),
                "low_apex": failures.get("low_apex_contact", 0),
                "ball_out": failures.get("ball_out_of_bounds", 0),
                "robot_body": failures.get("robot_body_contact", 0),
                "floor": failures.get("floor_contact", 0),
                "ball_speed": failures.get("ball_speed_limit", 0),
            }
        )
    return rows


def action_usage_rows() -> list[dict[str, object]]:
    rows = load_csv(
        "artifacts/ppo_runs/keep1_v34_17d_long_xyz012/analysis/"
        "keep1_v34_17d_long_xyz012_long7200_eval20_contacts.csv"
    )
    limits = [
        0.02,
        0.02,
        0.03,
        0.008,
        0.008,
        0.35,
        0.35,
        0.35,
        0.45,
        0.75,
        0.75,
        0.35,
        0.35,
        0.08,
        0.025,
        0.18,
        0.18,
    ]
    names = [
        "radial",
        "tangent",
        "z",
        "tilt_pitch",
        "tilt_roll",
        "vz_scale",
        "outgoing_x",
        "outgoing_y",
        "racket_vz",
        "trajectory_tilt_scale",
        "centering_tilt_scale",
        "racket_vx",
        "racket_vy",
        "target_apex_z",
        "strike_plane_z",
        "tracking_vx",
        "tracking_vy",
    ]
    result = []
    for index, name in enumerate(names):
        prefix = f"applied_action_{index}_"
        col = next(field for field in rows[0].keys() if field.startswith(prefix))
        vals = np.array([float(row[col]) for row in rows if row[col] != ""], dtype=float)
        abs_vals = np.abs(vals)
        result.append(
            {
                "index": index,
                "action": name,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "mean_abs": float(np.mean(abs_vals)),
                "limit": limits[index],
                "mean_abs_over_limit": float(np.mean(abs_vals) / limits[index]),
                "sat90_rate": float(np.mean(abs_vals > limits[index] * 0.9)),
            }
        )
    return result


def ablation_rows() -> list[dict[str, object]]:
    return [
        {"mask": "baseline", "mean_useful": 64.4167, "target_200_70": 8, "note": "original v34 policy"},
        {"mask": "drop tracking xy 15/16", "mean_useful": 50.1667, "target_200_70": 6, "note": "weak use, still helpful"},
        {"mask": "drop centering tilt 10", "mean_useful": 42.3333, "target_200_70": 4, "note": "low magnitude, high value"},
        {"mask": "drop weak bundle", "mean_useful": 59.1667, "target_200_70": 7, "note": "several weak axes together"},
        {"mask": "drop outgoing x 6", "mean_useful": 14.5, "target_200_70": 0, "note": "critical strong axis"},
        {"mask": "drop tilt roll 4", "mean_useful": 65.3333, "target_200_70": 8, "note": "strong magnitude, ambiguous value"},
        {"mask": "drop strike plane z 14", "mean_useful": 42.3333, "target_200_70": 2, "note": "critical height/timing axis"},
    ]


def plot_timeline(rows: list[dict[str, object]]) -> None:
    labels = [str(row["model"]) for row in rows]
    x = np.arange(len(rows))
    mean_useful = np.array([float(row["mean_useful"]) for row in rows])
    rate_30 = np.array([float(row["rate_30_plus"]) for row in rows]) * 100
    reset_xy = np.array([float(row["reset_xy_range"]) for row in rows])

    fig, ax1 = plt.subplots(figsize=(10.5, 5.4))
    bars = ax1.bar(x, mean_useful, color="#3f7f93", width=0.62, label="Mean useful bounces")
    ax1.set_ylabel("Mean useful bounces")
    ax1.set_ylim(0, max(mean_useful) * 1.25)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    for bar, row in zip(bars, rows):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"xy {float(row['reset_xy_range']):.3g}m\n{int(row['action_dim'])}D",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax2 = ax1.twinx()
    ax2.plot(x, rate_30, color="#d77a30", marker="o", linewidth=2.5, label="30+ rate")
    ax2.plot(x, reset_xy * 500, color="#6750a4", marker="s", linestyle="--", linewidth=1.8, label="Reset XY x500")
    ax2.set_ylabel("30+ rate (%) / reset XY x500")
    ax2.set_ylim(0, 100)
    ax1.set_title("Policy evolution: longer horizon, broader reset, then 17D expansion")
    ax1.grid(axis="y", alpha=0.25)
    lines, labels2 = ax1.get_legend_handles_labels()
    lines_b, labels_b = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines_b, labels2 + labels_b, loc="upper left")
    savefig("01_version_timeline_metrics.png")


def plot_failure_modes(rows: list[dict[str, object]]) -> None:
    labels = [str(row["model"]) for row in rows]
    failure_keys = [
        ("time_limit", "Time limit", "#668c4a"),
        ("low_apex", "Low apex", "#c76f54"),
        ("ball_out", "Ball out", "#d6a84f"),
        ("robot_body", "Body contact", "#7e6db0"),
        ("floor", "Floor", "#8a8a8a"),
        ("ball_speed", "Ball speed", "#4f80c0"),
    ]
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    plt.figure(figsize=(10.5, 5.2))
    for key, label, color in failure_keys:
        vals = np.array([float(row[key]) for row in rows])
        plt.bar(x, vals, bottom=bottom, label=label, color=color, width=0.62)
        bottom += vals
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Episodes")
    plt.title("Failure modes changed as the task became broader")
    plt.legend(ncol=3, fontsize=8)
    plt.grid(axis="y", alpha=0.25)
    savefig("02_failure_modes_by_version.png")


def plot_long_targets(rows: list[dict[str, object]]) -> None:
    labels = [str(row["model"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.34
    target_300 = [float(row["contacts300_useful100"]) for row in rows]
    target_400 = [float(row["contacts400_useful150"]) for row in rows]
    mean_useful = [float(row["mean_useful"]) for row in rows]
    fig, ax1 = plt.subplots(figsize=(9.5, 5.0))
    ax1.bar(x - width / 2, target_300, width, label="contacts>=300 & useful>=100", color="#3f7f93")
    ax1.bar(x + width / 2, target_400, width, label="contacts>=400 & useful>=150", color="#d77a30")
    ax1.set_ylabel("Episodes out of 20")
    ax1.set_ylim(0, 20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, mean_useful, color="#363636", marker="o", linewidth=2.2, label="Mean useful")
    ax2.set_ylabel("Mean useful bounces")
    ax2.set_ylim(0, max(mean_useful) * 1.25)
    lines, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Long-horizon target: v34 remains strongest after the v35 stability trade-off")
    savefig("03_long_horizon_target_hits.png")


def plot_apex_distribution() -> None:
    sources = [
        ("v33 long", "artifacts/ppo_runs/keep1_v33_17d_perf/analysis/keep1_v33_17d_perf_long7200_eval20_contacts.csv", "#c76f54"),
        ("v34 long", "artifacts/ppo_runs/keep1_v34_17d_long_xyz012/analysis/keep1_v34_17d_long_xyz012_long7200_eval20_contacts.csv", "#3f7f93"),
        ("v35 long", "artifacts/ppo_runs/keep1_v35_17d_strong_axis_stable/analysis/keep1_v35_17d_strong_axis_stable_long7200_eval20_contacts.csv", "#4f80c0"),
    ]
    plt.figure(figsize=(9.5, 5.2))
    bins = np.linspace(0.0, 0.55, 46)
    for label, path, color in sources:
        rows = load_csv(path)
        vals = [float(row["projected_contact_apex_height_above_racket"]) for row in rows if row["projected_contact_apex_height_above_racket"]]
        plt.hist(vals, bins=bins, alpha=0.42, label=label, density=True, color=color)
    plt.axvline(0.14, color="#7a1f1f", linestyle="--", linewidth=2, label="low-apex termination threshold 0.14m")
    plt.axvline(0.20, color="#333333", linestyle=":", linewidth=2, label="useful minimum 0.20m")
    plt.xlabel("Projected contact apex height above racket (m)")
    plt.ylabel("Density")
    plt.title("Low-apex tuning: allow recovery without counting low bounces as useful")
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.25)
    savefig("04_apex_height_distribution.png")


def plot_action_usage(rows: list[dict[str, object]]) -> None:
    labels = [f"{int(row['index'])}\n{row['action']}" for row in rows]
    vals = [float(row["mean_abs_over_limit"]) * 100 for row in rows]
    colors = []
    for row in rows:
        index = int(row["index"])
        if index in {6, 14}:
            colors.append("#c76f54")
        elif index in {8, 10, 12, 15, 16}:
            colors.append("#d6a84f")
        else:
            colors.append("#3f7f93")
    plt.figure(figsize=(13.0, 5.2))
    plt.bar(np.arange(len(rows)), vals, color=colors, width=0.72)
    plt.xticks(np.arange(len(rows)), labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("Mean abs action / limit (%)")
    plt.title("17D action usage in v34 contacts: magnitude is evidence, not proof of utility")
    plt.grid(axis="y", alpha=0.25)
    plt.axhline(10, color="#777777", linewidth=1, linestyle="--")
    savefig("05_action_usage_17d.png")


def plot_ablation(rows: list[dict[str, object]]) -> None:
    baseline = rows[0]["mean_useful"]
    labels = [row["mask"] for row in rows]
    vals = [float(row["mean_useful"]) for row in rows]
    colors = ["#3f7f93"] + ["#d6a84f", "#d6a84f", "#d6a84f", "#c76f54", "#7e6db0", "#c76f54"]
    plt.figure(figsize=(11.0, 5.4))
    bars = plt.bar(np.arange(len(rows)), vals, color=colors, width=0.68)
    plt.axhline(float(baseline), color="#333333", linestyle="--", linewidth=1.5, label="baseline")
    for bar, row in zip(bars, rows):
        delta = float(row["mean_useful"]) - float(baseline)
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{delta:+.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.xticks(np.arange(len(rows)), labels, rotation=28, ha="right")
    plt.ylabel("Mean useful bounces")
    plt.title("Action ablation: weak-looking axes can still carry value")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    savefig("06_action_ablation_mean_useful.png")


def plot_observation_action_diagram() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    ax.axis("off")
    groups = [
        ("Base robot/ball state\n35D", 0.08, 0.70, "#d7e8ec"),
        ("Phase\n4D", 0.08, 0.53, "#e8e1f2"),
        ("Contact context\n2D", 0.08, 0.40, "#e8e1f2"),
        ("Next intercept\n6D", 0.08, 0.27, "#e3efd4"),
        ("Desired outgoing\n3D", 0.08, 0.14, "#e3efd4"),
        ("Racket normal + tilt\n5D", 0.08, 0.01, "#f1e1d1"),
    ]
    for text, x, y, color in groups:
        ax.add_patch(plt.Rectangle((x, y), 0.23, 0.105, facecolor=color, edgecolor="#333333", linewidth=1.0))
        ax.text(x + 0.115, y + 0.052, text, ha="center", va="center", fontsize=10)
    ax.add_patch(plt.Rectangle((0.43, 0.36), 0.20, 0.22, facecolor="#f4f4f4", edgecolor="#333333", linewidth=1.2))
    ax.text(0.53, 0.47, "PPO policy\nMlpPolicy 64x64\ninput 55D", ha="center", va="center", fontsize=11, weight="bold")
    outputs = [
        ("Position + tilt\n5D", 0.75, 0.66, "#d7e8ec"),
        ("Outgoing velocity\n3D", 0.75, 0.51, "#f4d6d0"),
        ("Racket/tilt scale\n5D", 0.75, 0.36, "#f1e1d1"),
        ("Apex/timing\n2D", 0.75, 0.21, "#e3efd4"),
        ("Tracking residual\n2D", 0.75, 0.06, "#e8e1f2"),
    ]
    for text, x, y, color in outputs:
        ax.add_patch(plt.Rectangle((x, y), 0.20, 0.105, facecolor=color, edgecolor="#333333", linewidth=1.0))
        ax.text(x + 0.10, y + 0.052, text, ha="center", va="center", fontsize=10)
    for _, _, y, _ in groups:
        ax.annotate("", xy=(0.43, 0.47), xytext=(0.31, y + 0.052), arrowprops=dict(arrowstyle="->", color="#555555"))
    for _, x, y, _ in outputs:
        ax.annotate("", xy=(x, y + 0.052), xytext=(0.63, 0.47), arrowprops=dict(arrowstyle="->", color="#555555"))
    ax.text(0.53, 0.18, "Observation is not output.\n55D state -> 17D action residual.", ha="center", va="center", fontsize=11)
    savefig("07_observation_action_diagram.png")


def plot_monitor_curves() -> None:
    sources = [
        ("v33 monitor_004", "artifacts/ppo_runs/keep1_v33_17d_perf/monitor_004.monitor.csv", "#c76f54"),
        ("v34 monitor_001", "artifacts/ppo_runs/keep1_v34_17d_long_xyz012/monitor_001.monitor.csv", "#3f7f93"),
        ("v35 monitor_001", "artifacts/ppo_runs/keep1_v35_17d_strong_axis_stable/monitor_001.monitor.csv", "#4f80c0"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for label, path, color in sources:
        rows = []
        with (ROOT / path).open(encoding="utf-8") as handle:
            for raw in handle:
                if raw.startswith("#") or raw.startswith("r,"):
                    continue
                parts = raw.strip().split(",")
                if len(parts) < 3:
                    continue
                rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
        if not rows:
            continue
        rewards = np.array([row[0] for row in rows], dtype=float)
        lengths = np.array([row[1] for row in rows], dtype=float)
        steps = np.cumsum(lengths)
        window = min(50, max(5, len(rewards) // 10))
        kernel = np.ones(window) / window
        smooth_rewards = np.convolve(rewards, kernel, mode="valid")
        smooth_lengths = np.convolve(lengths, kernel, mode="valid")
        smooth_steps = steps[window - 1 :]
        axes[0].plot(smooth_steps, smooth_rewards, label=label, color=color)
        axes[1].plot(smooth_steps, smooth_lengths, label=label, color=color)
    axes[0].set_title("Monitor reward: log file shows training was alive")
    axes[0].set_xlabel("Cumulative env steps")
    axes[0].set_ylabel("Rolling episode return")
    axes[1].set_title("Episode length: longer survival emerges in logs")
    axes[1].set_xlabel("Cumulative env steps")
    axes[1].set_ylabel("Rolling episode length")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    savefig("08_monitor_training_curves.png")


def main() -> None:
    metrics = metric_rows()
    long_rows = long_eval_rows()
    usage = action_usage_rows()
    ablations = ablation_rows()

    write_csv(
        DATA_DIR / "version_metrics.csv",
        metrics,
        [
            "model",
            "action_dim",
            "reset_xy_range",
            "reset_height",
            "reset_velocity_xy_range",
            "reset_velocity_z_range",
            "evaluation_step_limit",
            "stable_cycle_reward_cap",
            "low_apex_grace",
            "episodes",
            "mean_useful",
            "max_useful",
            "rate_30_plus",
            "time_limit",
            "low_apex",
            "ball_out",
            "robot_body",
            "floor",
            "ball_speed",
        ],
    )
    write_csv(
        DATA_DIR / "long_horizon_metrics.csv",
        long_rows,
        [
            "model",
            "episodes",
            "mean_contacts",
            "mean_useful",
            "max_contacts",
            "max_useful",
            "contacts300_useful100",
            "contacts400_useful150",
            "time_limit",
            "low_apex",
            "ball_out",
            "robot_body",
            "floor",
            "ball_speed",
        ],
    )
    write_csv(
        DATA_DIR / "action_usage_v34.csv",
        usage,
        ["index", "action", "mean", "std", "mean_abs", "limit", "mean_abs_over_limit", "sat90_rate"],
    )
    write_csv(DATA_DIR / "action_ablation_v34.csv", ablations, ["mask", "mean_useful", "target_200_70", "note"])

    plot_timeline(metrics)
    plot_failure_modes(metrics)
    plot_long_targets(long_rows)
    plot_apex_distribution()
    plot_action_usage(usage)
    plot_ablation(ablations)
    plot_observation_action_diagram()
    plot_monitor_curves()


if __name__ == "__main__":
    main()
