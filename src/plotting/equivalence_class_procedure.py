# Standard imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.plotting.config import (
    FIG_AXIS_FONT_SIZE,
    FIG_HALF_SIZE,
    FIG_LEGEND_FONT_SIZE,
    FIG_TITLE_FONT_SIZE,
)

from matplotlib.patches import Rectangle

# Import function from `param_sweep.py` to run one experiment
from src.utils.param_sweep import run_experiment
from src.visualization.strategy import make_general_strategy_heatmap


# Import config functions for the Chain experiment
import src.param_sweeps.chain_world as chain_world
from src.utils.chain import make_chain_experiment, make_chain_transition


def run_chain(experiment, gammas: np.ndarray, probs: np.ndarray):
    h, w = experiment.height, experiment.width

    return run_experiment(
        experiment,
        transition_matrix_func=make_chain_transition,
        params=chain_world.default_params,
        gammas=gammas,
        probs=probs,
        start_state=chain_world.get_start_state(h, w),
    )


experiments = {
    "Chain": {
        "p2idx": {
            "Exercise": 0,
            "Disengage": 1,
        },
        "default": 1,
        "experiment": make_chain_experiment(
            width=5,
            disengage_prob=0.3,
            lost_progress_prob=0.1,
            goal_mag=9,
            disengage_reward=-1,
            burden=-1,
        ),
        "run_func": run_chain,
    },
}

if __name__ == "__main__":
    # Set up the parameters for the parameter sweep
    granularity = 25
    probs = np.linspace(0.4, 0.99, granularity)
    gammas = np.linspace(0.4, 0.99, granularity)

    run_only = set([])

    # Plot the dataframes
    pbar = tqdm(experiments.items())
    for name, data in pbar:
        if run_only and name not in run_only:
            continue

        pbar.set_description(f"Plotting {name} world")
        fig, ax = plt.subplots(figsize=FIG_HALF_SIZE, sharey=not True, sharex=True)
        # Adapt the p2idx names to something more descriptive
        p2idx = data["p2idx"]

        exp_res = data["run_func"](data["experiment"], gammas, probs)

        # Threshold the data to 0 and 1
        strat_data = np.where(exp_res.data > 0.5, 1, 0)

        # If default is not 0, then we need to flip the data
        if data.get("default", 0) != 0:
            strat_data = 1 - strat_data

        # Plot the heatmap
        make_general_strategy_heatmap(
            results=strat_data,
            probs=exp_res.probs,
            gammas=gammas,
            ax=ax,
            p2idx=p2idx,
            title="Equivalence Classification",
            annot=False,
            ax_labels=True,
            num_ticks=2,
            title_fontsize=FIG_TITLE_FONT_SIZE,
            legend_fontsize=FIG_LEGEND_FONT_SIZE,
            tick_fontsize=FIG_AXIS_FONT_SIZE,
            legend=False,
        )

        # Define the corner coordinates from bottom-left and around counter-clockwise
        corners = [
            (2, 2),
            (strat_data.shape[0] - 2, 2),
            (strat_data.shape[0] - 2, strat_data.shape[1] - 2),
            (2, strat_data.shape[1] - 2),
        ]
        labels = ["A", "B", "C", "D"]

        # Add a scatter plot and annotate each point
        for (x, y), label in zip(corners, labels):
            ax.scatter(
                x,
                y,
                s=800,
                edgecolor="black",
                linewidth=4,
                c="white",
            )
            ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=14,
                color="black",
                weight="bold",
            )

        # Add the arrow from 'A' to 'B' with the proper offset
        ax.arrow(
            corners[0][0] + 2,
            corners[0][1],
            corners[1][0] - corners[0][0] - 5,
            corners[1][1] - corners[0][1],
            head_width=1,
            head_length=1,
            fc="black",
            ec="black",
            lw=3,
        )

        # Add the arrow from 'B' to 'C' with the proper offset
        ax.arrow(
            corners[1][0],
            corners[1][1] + 2,
            corners[2][0] - corners[1][0],
            corners[2][1] - corners[1][1] - 5,
            head_width=1,
            head_length=1,
            fc="black",
            ec="black",
            lw=3,
        )

        # Add the arrow from 'C' to 'D' with the proper offset
        ax.arrow(
            corners[2][0] - 2,
            corners[2][1],
            corners[3][0] - corners[2][0] + 5,
            corners[3][1] - corners[2][1],
            head_width=1,
            head_length=1,
            fc="black",
            ec="black",
            lw=3,
        )

        # Add the last arrow from 'D' to 'A' with the proper offset
        ax.arrow(
            corners[-1][0],
            corners[-1][1] - 2,
            corners[0][0] - corners[-1][0],
            corners[0][1] - corners[-1][1] + 5,
            head_width=1,
            head_length=1,
            fc="black",
            ec="black",
            lw=3,
        )

        # Define the mid-points of the edges for placement of the squares
        midpoints = [
            (strat_data.shape[0] / 2, 0),
            (strat_data.shape[0] - 1, strat_data.shape[1] / 2),
            (strat_data.shape[0] / 2, strat_data.shape[1] - 1),
            (0, strat_data.shape[1] / 2),
        ]
        numbers = [1, 0, 1, 0]
        offsets = [(0.5, 4), (-3, 0), (-1, -3), (4, 0)]

        # Size of the square, adjust as needed
        square_size = 3

        for (x, y), number, (dx, dy) in zip(midpoints, numbers, offsets):
            # Calculate the new coordinates after applying the padding
            x_new = x + dx
            y_new = y + dy

            # Create a square as a rectangle with equal width and height
            square = Rectangle(
                (x_new - square_size / 2, y_new - square_size / 2),
                square_size,
                square_size,
                fill=True,
                color="white",
            )
            border_square = Rectangle(
                (x_new - square_size / 2, y_new - square_size / 2),
                square_size,
                square_size,
                fill=False,
                edgecolor="black",
                linewidth=2,
            )

            # Add the square to the plot
            ax.add_patch(square)
            ax.add_patch(border_square)

            # Add the number to the square
            ax.text(
                x_new,
                y_new,
                str(number),
                ha="center",
                va="center",
                fontsize=14,
                color="black",
                weight="bold",
            )

        # Make it so that the figure and the axes labels are not cut off
        plt.tight_layout()

        # plt.show()

        # Save the figure
        fig.savefig(
            f"images/plots/equivalence_class_procedure.pdf",
            dpi=300,
            bbox_inches="tight",
        )
