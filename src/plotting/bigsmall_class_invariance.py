import pickle

from matplotlib import pyplot as plt
import numpy as np
from src.plotting.config import (
    FIG_AXIS_FONT_SIZE,
    FIG_HALF_SIZE,
    FIG_LEGEND_FONT_SIZE,
    FIG_TITLE_FONT_SIZE,
)

from src.visualization.strategy import make_general_strategy_heatmap


def load_data(filename):
    return pickle.load(
        open("local_images/small_and_big_reward_world/" + filename, "rb")
    )


def unpack_data(data):
    strategy_data = data["strategy_data"]
    rows, cols = data["grid_dimensions"]
    search_parameters = data["search_parameters"]
    default_params = data["default_params"]
    probs = data["probs"]
    gammas = data["gammas"]

    return strategy_data, rows, cols, search_parameters, default_params, probs, gammas


if __name__ == "__main__":
    data = load_data("2023-04-27_18-32-26_metadata.pkl")

    (
        strategy_data,
        rows,
        cols,
        search_parameters,
        default_params,
        probs,
        gammas,
    ) = unpack_data(data)

    indices = [9, 15, 27, 31]

    for num, i in enumerate(indices, start=1):
        fig, ax = plt.subplots(figsize=FIG_HALF_SIZE)

        default = {
            "width": default_params["width"],
            "small_reward_frac": default_params["small_reward_frac"],
            strategy_data[i][2]: strategy_data[i][3],
        }

        # Rename the keys to be more readable
        default["Width"] = default.pop("width")
        default["Reward ratio"] = default.pop("small_reward_frac")

        # Sort the keys in default
        default = dict(sorted(default.items()))

        # Format `default` as a nice string for the title
        title = ", ".join([f"{k}={v}" for k, v in default.items()])

        make_general_strategy_heatmap(
            results=np.where(strategy_data[i][0] > 0.5, 1, 0),
            probs=probs,
            gammas=gammas,
            ax=ax,
            p2idx={
                "Close small R": 0,
                "Far large R": 1,
            },
            title=title,
            annot=False,
            ax_labels=True,
            num_ticks=3,
            title_fontsize=FIG_TITLE_FONT_SIZE,
            legend_fontsize=FIG_LEGEND_FONT_SIZE,
            tick_fontsize=FIG_AXIS_FONT_SIZE,
        )

        # Save the figure
        plt.tight_layout()
        plt.savefig(
            f"images/plots/big_small_invariance_{num}.pdf",
            bbox_inches="tight",
            dpi=300,
        )
