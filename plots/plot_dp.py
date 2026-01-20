import matplotlib.pyplot as plt
import sys
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from py_utils import create_color_map, create_marker_map, format_bytes
from parser import *

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

PROTOCOL = ["Simple", "LL", "LL128"]
ALGO = ["ring", "tree", "nvls", "collnet"]
THREADS = ["64","128", "256", "512"]
CHANNELS = ["2", "4", "8", "16", "32"]


def plot_runtime_scaling(df, model_name, bucket_size=None, local_batch_size=None, runs_per_rank=50, networks=["ib", "eth"], colors=None, networks_labels=None):
    """
    Plot runtime per step vs world_size for data parallel scaling. Ideal scaling is a flat line starting from the first world_size.
    """
    if colors is None:
        colors = {"ib": "orange", "eth": "blue"}
    if networks_labels is None:
        networks_labels = {net: net for net in networks}
    
    filtered_df = df[df["model_name"] == model_name]
    if bucket_size is not None:
        filtered_df = filtered_df[filtered_df["num_buckets"] == bucket_size]

    plt.figure(figsize=(10,6))

    for net in networks:
        job_df = filtered_df[filtered_df["network"] == net].copy()
        if job_df.empty:
            continue  # skip networks with no data

        # runtime medio per step aggregando su rank e run
        job_df["run_index"] = job_df.groupby("rank").cumcount() % runs_per_rank
        agg_df = job_df.groupby(["world_size", "run_index"])["runtime_s"].mean().reset_index()
        mean_runtime = agg_df.groupby("world_size")["runtime_s"].mean()

        # ordinamento per sicurezza
        ws = np.array(sorted(mean_runtime.index))
        rt = mean_runtime.loc[ws].values

        label_net = networks_labels.get(net, net)
        plt.plot(ws, rt, "o-", color=colors.get(net, "gray"), label=label_net)

        # linea ideale: runtime costante al valore del primo world_size
        # ideal_runtime = np.full(ws.shape, rt[0], dtype=float)
        # # plt.plot(ws, ideal_runtime, "--", color=colors.get(net, "gray"), alpha=0.5, label="Ideal" if net==networks[0] else None)

    plt.xlabel("World Size")
    plt.ylabel("Time (s)")
    plt.title(f"Data Parallel Scaling - Model: {model_name}, Local Batch: {local_batch_size}")
    plt.xticks(ws)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    #save png to file
    output_dir = Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    bucket_str = f"_b{bucket_size}" if bucket_size is not None else ""
    output_path = output_dir / f"dp_scaling_{model_name}{bucket_str}_lbs{local_batch_size}.png"
    plt.savefig(output_path)


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

if __name__ == "__main__":
    df = get_metrics_dataframe('dp')
    colors = create_color_map(["ib", "eth", "boost_usr_prod"])

    networks_labels = {
        "boost_usr_prod": "boost_usr_prod",
    }
    