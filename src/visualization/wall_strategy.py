import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns


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
    # set the number of tick labels to display
    num_ticks = 10

    # compute the indices to use for the tick labels
    gamma_indices = np.round(np.linspace(0, len(gammas) - 1, num_ticks)).astype(int)
    prob_indices = np.round(np.linspace(0, len(probs) - 1, num_ticks)).astype(int)

    # create the tick labels
    gamma_ticks = [round(gammas[i], 2) for i in gamma_indices]
    prob_ticks = [round(probs[i], 2) for i in prob_indices]

    # plot the heatmap
    ax = sns.heatmap(results, annot=annot, cmap="Blues", fmt="d", ax=ax, cbar=False)

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

    ax.set_title(title or "Optimal strategy (1: Right, 3: Down)")

    # Set legend to the right to explain the numbers 1 and 3 with same colors as the heatmap
    if legend:
        ax.legend(
            handles=[
                mpatches.Patch(color="white", label="1: Right"),
                mpatches.Patch(color="darkblue", label="3: Down"),
            ],
        )