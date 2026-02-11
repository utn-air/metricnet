import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pub_style = {
    "figure.dpi": 300,
    "figure.figsize": (3.5, 2.5),
    "font.size": 8,
    "font.family": "sans-serif",
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "axes.labelpad": 4,
    "axes.grid": True,
    "grid.color": "0.85",
    "grid.linewidth": 0.2,
    "grid.linestyle": "--",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1,
    "lines.markersize": 4,
    "boxplot.boxprops.color": "black",
    "boxplot.capprops.color": "black",
    "boxplot.whiskerprops.color": "black",
    "boxplot.medianprops.color": "#000000",
    "savefig.bbox": "tight",
    "savefig.transparent": False,
}
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update(pub_style)


def better_labels(label: list[str]) -> list[str]:
    better_labels = [txt.replace("_", " ").replace("-", " ") for txt in label]
    better_labels = [txt.replace("flownav", "FlowNav") for txt in better_labels]
    better_labels = [txt.replace("nomad", "NoMaD") for txt in better_labels]
    better_labels = [txt.replace("no depth", "w/o depth") for txt in better_labels]
    better_labels = [
        txt.replace("depth", "w/ depth") if "w/o" not in txt else txt
        for txt in better_labels
    ]
    better_labels = [txt.replace("metricnet", "w/ scale") for txt in better_labels]
    better_labels = [
        txt.replace("navidiffusor", "w/ NaviDiffusor") for txt in better_labels
    ]
    return better_labels


def load_data(base_path, eval_type="nav"):
    records = []
    base_dir = Path(base_path)
    for scene_dir in base_dir.iterdir():
        if not scene_dir.is_dir():
            continue
        scene = scene_dir.name
        for agent_dir in scene_dir.iterdir():
            if not agent_dir.is_dir():
                continue
            agent = agent_dir.name
            for json_file in agent_dir.glob("*.json"):
                algo = json_file.stem.split("_")[1:]
                algo = "_".join(algo) if algo else "unknown"
                try:
                    data = json.loads(json_file.read_text())
                except Exception as e:
                    print(f"Warning: failed to load {json_file}: {e}")
                    continue
                if eval_type == "nav":
                    gr = data.get("goal_reached", 0)
                    mg = data.get("max_goal", 1) or 1
                    pct = (gr / mg) * 100
                    records.append(
                        {
                            "scene": scene,
                            "agent": agent,
                            "algorithm": algo,
                            "goal_reached": gr,
                            "max_goal": mg,
                            "percentage": pct,
                        }
                    )
                elif eval_type == "exp":
                    dist_explored = data.get("distance_explored", 0)
                    valid_steps = data.get("steps_until_collision", 1) or 1
                    records.append(
                        {
                            "scene": scene,
                            "agent": agent,
                            "algorithm": algo,
                            "distance_explored": dist_explored,
                            "valid_steps": valid_steps,
                        }
                    )
                else:
                    raise ValueError(f"Unknown eval_type: {eval_type}")

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)

    if eval_type == "nav":
        df = (
            df.groupby(["scene", "agent", "algorithm"])
            .agg(
                {
                    "goal_reached": "mean",
                    "max_goal": "mean",
                    "percentage": "mean",
                }
            )
            .reset_index()
        )
    elif eval_type == "exp":
        df = (
            df.groupby(["scene", "agent", "algorithm"])
            .agg(
                {
                    "distance_explored": "mean",
                    "valid_steps": "mean",
                }
            )
            .reset_index()
        )
    else:
        raise ValueError(f"Unknown eval_type: {eval_type}")
    return df


def plot_mean_std_nav(df, save_path="temp.png"):
    summary = df.groupby("algorithm")["percentage"].agg(["mean", "std"]).reset_index()
    max_vals = df.groupby("algorithm")["percentage"].max().reset_index()
    plt.figure(figsize=(8, 5))
    labels = better_labels(summary["algorithm"].tolist())
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    bar_colors = [colors[i % len(colors)] for i in range(len(labels))]
    ax = plt.bar(
        labels, summary["mean"], yerr=summary["std"], capsize=5, color=bar_colors
    )
    plt.ylabel("Average % Goal Reached")
    plt.xlabel("")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 100)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, color="0.85", zorder=0)
    for idx, (bar, mean) in enumerate(zip(ax.patches, summary["mean"])):
        bar.set_edgecolor("#000000")
        bar.set_linewidth(0.8)
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() - 0.02,
            height + 1,
            f"{mean:.1f}",
            ha="right",
            va="bottom",
            fontsize=8,
        )
        algo = summary["algorithm"].iloc[idx]
        max_val = max_vals[max_vals["algorithm"] == algo]["percentage"].values[0]
        x_center = bar.get_x() + bar.get_width() / 2
        y_marker_max = max_val + 2
        plt.plot(
            x_center, y_marker_max, marker="v", color="black", markersize=7, zorder=5
        )
        plt.text(
            x_center,
            y_marker_max - 3,
            f"{max_val:.1f}",
            ha="center",
            va="top",
            fontsize=7,
            color="black",
        )
    plt.savefig(save_path)


def plot_mean_std_exp(df, save_path="temp.png"):
    summary = (
        df.groupby("algorithm")["distance_explored"].agg(["mean", "std"]).reset_index()
    )
    max_vals = df.groupby("algorithm")["distance_explored"].max().reset_index()
    plt.figure(figsize=(8, 5))
    labels = better_labels(summary["algorithm"].tolist())
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    bar_colors = [colors[i % len(colors)] for i in range(len(labels))]
    ax = plt.bar(
        labels, summary["mean"], yerr=summary["std"], capsize=5, color=bar_colors
    )
    plt.ylabel("Average Distance Explored (m)")
    plt.xlabel("")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, color="0.85", zorder=0)
    for idx, (bar, mean) in enumerate(zip(ax.patches, summary["mean"])):
        bar.set_edgecolor("#000000")
        bar.set_linewidth(0.8)
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() - 0.02,
            height + 0.2,
            f"{mean:.1f}",
            ha="right",
            va="bottom",
            fontsize=8,
        )
        algo = summary["algorithm"].iloc[idx]
        max_val = max_vals[max_vals["algorithm"] == algo]["distance_explored"].values[0]
        x_center = bar.get_x() + bar.get_width() / 2
        y_marker_max = max_val + 2
        plt.plot(
            x_center, y_marker_max, marker="v", color="black", markersize=7, zorder=5
        )
        plt.text(
            x_center,
            y_marker_max - 0.5,
            f"{max_val:.1f}",
            ha="center",
            va="top",
            fontsize=7,
            color="black",
        )
    plt.savefig(save_path)


def plot_distribution_nav(df, save_path="temp.png"):
    filtered_df = df.groupby("algorithm").percentage.apply(np.array).reset_index()
    values = filtered_df["percentage"].values
    labels = filtered_df["algorithm"].values
    labels = better_labels(labels.tolist())
    plt.figure(figsize=(6, 3))
    ax = plt.boxplot(values, patch_artist=True, tick_labels=labels)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for patch, flier, color in zip(ax["boxes"], ax["fliers"], colors):
        patch.set_facecolor(color)
        flier.set(marker="o", markerfacecolor=color)
    plt.ylabel("% Goal Reached")
    plt.axis(ymin=0, ymax=100)
    plt.xticks(rotation=45, ha="right")
    plt.savefig(save_path)


def plot_distribution_exp(df, save_path="temp.png"):
    filtered_df = (
        df.groupby("algorithm").distance_explored.apply(np.array).reset_index()
    )
    values = filtered_df["distance_explored"].values
    labels = filtered_df["algorithm"].values
    labels = better_labels(labels.tolist())
    plt.figure(figsize=(6, 3))
    ax = plt.boxplot(values, patch_artist=True, tick_labels=labels)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for patch, flier, color in zip(ax["boxes"], ax["fliers"], colors):
        patch.set_facecolor(color)
        flier.set(marker="o", markerfacecolor=color)
    plt.ylabel("% Goal Reached")
    plt.xticks(rotation=45, ha="right")
    plt.savefig(save_path)


def plot_results(df, save_dir, eval_type="nav"):
    if eval_type == "nav":
        plot_mean_std_nav(df, save_dir / "average_nav.png")
        plot_distribution_nav(df, save_dir / "distribution_nav.png")
    elif eval_type == "exp":
        plot_mean_std_exp(df, save_dir / "average_exp.png")
        plot_distribution_exp(df, save_dir / "distribution_exp.png")
    else:
        raise ValueError(f"Unknown eval_type: {eval_type}")
