import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns


cmap = sns.color_palette("tab10")


def make_general_strategy_heatmap(
    *,
    results,
    probs,
    gammas,
    ax,
    p2idx,
    title=None,
    annot=True,
    ax_labels=True,
    num_ticks=10,
    title_fontsize=8,
    legend_fontsize=5,
    tick_fontsize=8,
) -> None:
    make_strategy_heatmap(
        results,
        probs,
        gammas,
        ax,
        title=title,
        annot=annot,
        ax_labels=ax_labels,
        num_ticks=num_ticks,
        title_fontsize=title_fontsize,
        tick_fontsize=tick_fontsize,
    )

    # Create legend patches
    legend_patches = [
        mpatches.Patch(color=cmap[idx], label=f"{path}") for path, idx in p2idx.items()  # type: ignore
    ]

    ax.legend(handles=legend_patches, loc="upper left", fontsize=legend_fontsize)


def make_strategy_heatmap(
    results,
    probs,
    gammas,
    ax,
    title=None,
    annot=True,
    ax_labels=True,
    num_ticks=10,
    title_fontsize=8,
    tick_fontsize=8,
) -> None:
    # compute the indices to use for the tick labels
    gamma_indices = np.round(np.linspace(0, len(gammas) - 1, num_ticks)).astype(int)
    prob_indices = np.round(np.linspace(0, len(probs) - 1, num_ticks)).astype(int)

    # create the tick labels
    gamma_ticks = [round(gammas[i], 2) for i in gamma_indices]
    prob_ticks = [round(probs[i], 2) for i in prob_indices]

    # plot the heatmap
    sns.heatmap(
        results,
        annot=annot,
        cmap=cmap,
        fmt="d",
        ax=ax,
        cbar=False,
        square=True,
        vmax=10,
    )

    # set the tick labels and positions
    ax.xaxis.set_major_locator(ticker.FixedLocator(gamma_indices))
    ax.set_xticklabels(gamma_ticks, rotation=0, size=tick_fontsize)
    ax.yaxis.set_major_locator(ticker.FixedLocator(prob_indices))
    ax.set_yticklabels(prob_ticks, rotation=90, size=tick_fontsize)

    # invert the y-axis
    ax.invert_yaxis()

    if ax_labels:
        ax.set_ylabel(r"Confidence level $p_u$", size=tick_fontsize)
        ax.set_xlabel(r"Discount factor $\gamma$", size=tick_fontsize)

    if title:
        ax.set_title(title, size=title_fontsize)

    return ax
