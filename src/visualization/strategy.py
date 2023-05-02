import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns


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
    )

    # Obtain the colormap
    cmap = sns.color_palette("Blues", len(p2idx))

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
        cmap="Blues",
        fmt="d",
        ax=ax,
        cbar=False,
        square=True,
        vmin=-1,
        vmax=np.max(results),
    )

    # set the tick labels and positions
    ax.xaxis.set_major_locator(ticker.FixedLocator(gamma_indices))
    ax.set_xticklabels(gamma_ticks, rotation=90, size=8)
    ax.yaxis.set_major_locator(ticker.FixedLocator(prob_indices))
    ax.set_yticklabels(prob_ticks, size=8, rotation=0)

    # invert the y-axis
    ax.invert_yaxis()

    if ax_labels:
        ax.set_xlabel("Gamma")
        ax.set_ylabel("Confidence")

    if title:
        ax.set_title(title, size=title_fontsize)

    return ax


def plot_wall_strategy_heatmap(
    results,
    probs,
    gammas,
    ax,
    title=None,
    legend=True,
    annot=True,
    ax_labels=True,
):
    make_strategy_heatmap(
        results=results,
        probs=probs,
        gammas=gammas,
        ax=ax,
        title=title,
        annot=annot,
        ax_labels=ax_labels,
    )

    # Set legend to the right to explain the numbers 1 and 3 with same colors as the heatmap
    if legend:
        ax.legend(
            handles=[
                mpatches.Patch(color="white", label="1: Right"),
                mpatches.Patch(color="darkblue", label="3: Down"),
            ],
        )
