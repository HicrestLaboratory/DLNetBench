import json
import pandas as pd
import matplotlib.pyplot as plt
import sbatchman as sbm
import sys
import numpy as np
import os
from scipy.stats import gmean 
from pathlib import Path
import sys

project_root = Path().resolve().parent  # adjust .parent / .parent.parent as needed
sys.path.append(str(project_root))
from py_utils.utils import create_color_map, create_marker_map, format_bytes

from parser_dp import *

FONT_TITLE = 18
FONT_AXES = 18
FONT_TICKS = 16
FONT_LEGEND = 12

plt.rc('axes', titlesize=FONT_AXES)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_AXES)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONT_TICKS)   # fontsize of the tick labels
plt.rc('ytick', labelsize=FONT_TICKS)   # fontsize of the tick labels
plt.rc('legend', fontsize=FONT_LEGEND)  # legend fontsize
plt.rc('figure', titlesize=FONT_TITLE)  # fontsize of the figure title


def plot_barrier_scatter_by_bucket(df, model_name, world_size, networks=["ib", "eth"], colors=None, networks_labels=None, runs_per_rank=50):
    """
    Scatter plot of barrier times per run, aggregated across ranks.
    Runs are assumed to be ordered and equal across ranks.
    """
    if colors is None:
        colors = {"ib": "orange", "eth": "blue"}
    
    filtered_df = df[df["model_name"] == model_name]
    buckets = sorted(filtered_df["num_buckets"].unique())

    # Compute MiB labels for x-axis
    bucket_sizes_kib = []
    for b in buckets:
        mean_bytes = filtered_df[filtered_df["num_buckets"] == b]["msg_size_avg_bytes"].mean()
        mib = format_bytes(mean_bytes, binary=True)
        bucket_sizes_kib.append(f"{b} buckets\n{mib}")

    plt.figure(figsize=(10,6))

    for i, net in enumerate(networks):
        job_df = filtered_df[
            (filtered_df["network"] == net) &
            (filtered_df["world_size"] == world_size)
        ].copy()

        # Assign run index manually per rank
        job_df["run_index"] = job_df.groupby("rank").cumcount() % runs_per_rank

        # Aggregate across ranks per run
        agg_df = (
            job_df.groupby(["num_buckets", "run_index"])["barrier_time_s"]
                  .mean()
                  .reset_index()
        )

        label_net = networks_labels[net]

        for j, b in enumerate(buckets):
            runs = agg_df[agg_df["num_buckets"] == b]["barrier_time_s"].values
            # horizontal position: bucket index + network offset + small jitter
            x_positions = np.full_like(runs, j) + (i - 0.5) * 0.2 + (np.random.rand(len(runs)) - 0.5) * 0.05
            plt.scatter(
                x_positions, runs,
                color=colors.get(net, "gray"),
                alpha=0.7,
                label=label_net if j==0 else ""
            )

    plt.xticks(np.arange(len(buckets)), bucket_sizes_kib)
    plt.xlabel("Buckets (Msg size x bucket)")
    plt.ylabel("Barrier Time (s)")
    plt.title(f"Barrier Time Distribution\nModel: {model_name}, World Size: {world_size}")
    
    # Remove duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    #save png to file
    output_dir = Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"barrier_scatter_{model_name}_ws{world_size}.png"
    plt.savefig(output_path)
    plt.show()


if __name__ == "__main__":
    df = get_dp_dataframe()
    colors = create_color_map(["ib", "eth", "boost_usr_prod"])

    networks_labels = {
        "ib": "HAICGU-ib",
        "eth": "HAICGU-eth",
        "boost_usr_prod": "boost_usr_prod",
    }


    plot_barrier_scatter_by_bucket(
        df,                     # your DataFrame
        model_name="vit_h_16_128",
        world_size=4,            # number of ranks/nodes
        networks=["ib", "eth", "boost_usr_prod"],  # networks to plot
        colors=colors,
        networks_labels=networks_labels
    )