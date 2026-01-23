import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from paretoset import paretoset
from matplotlib.lines import Line2D

# Local utils
sys.path.append(str(Path(__file__).parent.parent))
from py_utils import format_bytes

# =========================
# Global style
# =========================

FONT_TITLE = 18
FONT_AXES = 16
FONT_TICKS = 14
FONT_LEGEND = 12

plt.rc('axes', titlesize=FONT_AXES, labelsize=FONT_AXES)
plt.rc('xtick', labelsize=FONT_TICKS)
plt.rc('ytick', labelsize=FONT_TICKS)
plt.rc('legend', fontsize=FONT_LEGEND)
plt.rc('figure', titlesize=FONT_TITLE)

sns.set_theme()


# =========================
# Data processing
# =========================

def aggregate_by_configuration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate runtime and energy per configuration.
    - Runtime: averaged across ranks and runs (parallel execution)
    - Energy: summed across ranks, then averaged across runs (total system energy)
    """
    agg = (
        df.groupby(
            ['model_name', 'protocol', 'algorithm', 'threads', 'channels'],
            as_index=False
        )
        .agg(
            runtime=('runtime', 'mean'),
            energy_consumed=('energy_consumed', 'sum'),  # Changed from 'mean' to 'sum'
            num_measurements=('rank', 'count')
        )
    )

    agg['Protocol x Algorithm'] = agg['protocol'] + ' x ' + agg['algorithm']
    agg['Threads x Channels'] = (
        'T' + agg['threads'].astype(str) +
        ' x C' + agg['channels'].astype(str)
    )

    return agg


def compute_pareto_frontier(
    df: pd.DataFrame,
    x_col: str = 'energy_consumed',
    y_col: str = 'runtime'
) -> pd.DataFrame:
    """
    Compute Pareto frontier (both objectives minimized).
    """
    mask = paretoset(
        df[[x_col, y_col]],
        sense=["min", "min"]
    )
    return df[mask].sort_values(by=x_col).copy()


# =========================
# Plotting helpers
# =========================

def draw_pareto_frontier(ax, pareto_df, x_col, y_col, color='red'):
    """
    Draw staircase-style Pareto frontier.
    """
    if len(pareto_df) < 2:
        return

    pts = pareto_df[[x_col, y_col]].values
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        ax.plot([x0, x1], [y0, y0], color=color, lw=2.5, zorder=10)
        ax.plot([x1, x1], [y0, y1], color=color, lw=2.5, zorder=10)


def pretty_model_name(model: str) -> str:
    mapping = {
        'vit_h_16_128': 'ViT-H',
        'vit_l_16_128': 'ViT-L'
    }
    return mapping.get(model.lower(), model)


# =========================
# Main plotting function
# =========================

def plot_energy_runtime_per_model(df: pd.DataFrame, save_path: str | None = None):
    """
    Side-by-side Energy vs Runtime plots (one per model)
    with Pareto frontier.
    """
    agg_df = aggregate_by_configuration(df)
    models = sorted(agg_df['model_name'].unique())

    if len(models) != 2:
        print(f"Warning: expected 2 models, found {len(models)}: {models}")

    fig, axes = plt.subplots(1, len(models), figsize=(20, 8))
    if len(models) == 1:
        axes = [axes]

    protocols = sorted(agg_df['Protocol x Algorithm'].unique())
    thread_channels = sorted(agg_df['Threads x Channels'].unique())

    colors = dict(zip(protocols, sns.color_palette("tab20", len(protocols))))
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'D', 'P', 'X']
    marker_map = dict(
        zip(
            thread_channels,
            markers * (len(thread_channels) // len(markers) + 1)
        )
    )

    for ax, model in zip(axes, models):
        model_df = agg_df[agg_df['model_name'] == model]

        # Message size (assumed constant per model)
        msg_size = df[df['model_name'] == model]['msg_size_avg_bytes'].iloc[0]
        msg_size_str = format_bytes(msg_size)

        pareto_df = compute_pareto_frontier(model_df)

        for prot in protocols:
            for tc in thread_channels:
                subset = model_df[
                    (model_df['Protocol x Algorithm'] == prot) &
                    (model_df['Threads x Channels'] == tc)
                ]
                if subset.empty:
                    continue

                ax.scatter(
                    subset['energy_consumed'],
                    subset['runtime'],
                    color=colors[prot],
                    marker=marker_map[tc],
                    s=140,
                    edgecolor='black',
                    linewidth=1,
                    alpha=0.75,
                    zorder=5
                )

        draw_pareto_frontier(ax, pareto_df, 'energy_consumed', 'runtime')

        ax.set_xlabel('Energy Consumed (J)')
        ax.set_ylabel('Runtime (s)')
        ax.set_title(
            f"{pretty_model_name(model)} — Energy vs Runtime\n"
            f"Message Size: {msg_size_str}"
        )
        ax.grid(True, alpha=0.3)

    # =========================
    # Shared legend
    # =========================

    legend_items = [
        Line2D([0], [0], color='red', lw=2.5, label='Pareto Frontier'),
        Line2D([0], [0], linestyle='none', label='')
    ]

    legend_items.append(
        Line2D([0], [0], linestyle='none', label='Protocol x Algorithm:')
    )
    for p in protocols:
        legend_items.append(
            Line2D(
                [0], [0],
                marker='o',
                color='w',
                markerfacecolor=colors[p],
                markeredgecolor='black',
                markersize=8,
                label=f'  {p}'
            )
        )

    legend_items.append(
        Line2D([0], [0], linestyle='none', label='')
    )
    legend_items.append(
        Line2D([0], [0], linestyle='none', label='Threads x Channels:')
    )

    for tc in thread_channels:
        legend_items.append(
            Line2D(
                [0], [0],
                marker=marker_map[tc],
                color='w',
                markerfacecolor='gray',
                markeredgecolor='black',
                markersize=8,
                label=f'  {tc}'
            )
        )

    fig.legend(
        handles=legend_items,
        loc='center right',
        bbox_to_anchor=(1.12, 0.5),
        frameon=True
    )

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {save_path}")
    else:
        return fig


# =========================
# Entry point
# =========================

if __name__ == "__main__":
    df = pd.read_csv("dp_leonardo_intra.csv")
    plot_energy_runtime_per_model(df, save_path="energy_runtime_comparison.png")
