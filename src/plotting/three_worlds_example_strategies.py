# Standard imports
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from src.plotting.config import FIG_FULL_SIZE, FIG_TITLE_FONT_SIZE

# Import function from `param_sweep.py` to run one experiment
from src.utils.param_sweep import ExperimentResult, run_experiment
from src.utils.pessimism import scale_from_pessimism
from src.utils.small_big import make_smallbig_experiment
from src.utils.transition_matrix import id_func

# Import config functions for the wall experiment
import src.param_sweeps.wall_world as wall_world

# Import config functions for the cliff experiment
import src.param_sweeps.cliff_world as cliff_world
from src.utils.cliff import make_cliff_experiment, make_cliff_transition

# Import config functions for the smallbig experiment
import src.param_sweeps.smallbig_world as smallbig_world
from src.utils.wall import make_wall_experiment
from src.visualization.strategy import make_general_strategy_heatmap


def run_wall(experiment, gammas: np.ndarray, scalers: np.ndarray):
    h, w = experiment.height, experiment.width

    return run_experiment(
        experiment,
        transition_matrix_func=id_func,
        params=wall_world.default_params,
        gammas=gammas,
        probs=scalers,
        start_state=wall_world.get_start_state(h, w),
        realized_probs_indices=wall_world.get_realized_probs_indices(h, w),
    )


def run_cliff(experiment, gammas: np.ndarray, probs: np.ndarray):
    h, w = experiment.height, experiment.width

    return run_experiment(
        experiment,
        transition_matrix_func=make_cliff_transition,
        params=cliff_world.default_params,
        gammas=gammas,
        probs=probs,
        start_state=cliff_world.get_start_state(h, w),
    )


def run_smallbig(experiment, gammas: np.ndarray, probs: np.ndarray):
    h, w = experiment.height, experiment.width

    return run_experiment(
        experiment,
        transition_matrix_func=id_func,
        params=smallbig_world.default_params,
        gammas=gammas,
        probs=probs,
        start_state=smallbig_world.get_start_state(h, w),
    )


strategy_names = {
    "Wall": {
        "Around": 0,
        "Through": 1,
    },
    "Cliff": {
        "Hug cliff": 0,
        "Keep space": 1,
    },
    "Smallbig": {
        "Small reward": 0,
        "Big reward": 1,
    },
}


if __name__ == "__main__":
    # Set up the gridworlds
    height, width = 5, 5
    print("Creating gridworlds...")
    wall, cliff, smallbig = [
        make_wall_experiment(
            height=2,
            width=5,
            reward_mag=300,
            neg_mag=-10,
            latent_cost=0,
        ),
        make_cliff_experiment(
            height=5,
            width=11,
            reward_mag=100,
            small_r_mag=0,
        ),
        make_smallbig_experiment(
            height=7,
            width=7,
            big_reward=100,
            small_reward_frac=0.2,
        ),
    ]

    # Set up the parameters for the parameter sweep
    granularity = 30
    probs = np.linspace(0.3, 0.999, granularity)
    gammas = np.linspace(0.4, 0.99, granularity)
    scalers = 2 ** np.linspace(4.2, -2.5, granularity)

    # Run the experiment for each gridworld
    results: Dict[str, ExperimentResult] = {}

    print("Running experiments...")
    results["Wall"] = run_wall(wall, gammas, scalers)
    print("Wall done.")
    results["Cliff"] = run_cliff(cliff, gammas, probs)
    print("Cliff done.")
    results["Smallbig"] = run_smallbig(smallbig, gammas, probs)
    print("Smallbig done.")

    # Plot the dataframes
    fig, axes = plt.subplots(1, 3, figsize=FIG_FULL_SIZE, sharey=not True, sharex=True)
    for ax, (name, exp_res), letter in zip(
        axes, results.items(), ["(a)", "(b)", "(c)"]
    ):
        # Adapt the p2idx names to something more descriptive
        p2idx = strategy_names[name]

        # Threshold the data to 0 and 1
        data = np.where(exp_res.data > 0.5, 1, 0)

        # Plot the heatmap
        make_general_strategy_heatmap(
            results=data,
            probs=exp_res.probs,
            gammas=gammas,
            ax=ax,
            p2idx=p2idx,
            title=f"{letter} {name}",
            annot=False,
            ax_labels=False,
            num_ticks=10,
            title_fontsize=FIG_TITLE_FONT_SIZE,
            legend_fontsize=8,
        )

    # Set the axis labels
    axes[0].set_ylabel(r"Confidence level $p_u$")
    for ax in axes:
        ax.set_xlabel(r"Discount factor $\gamma$")

    # Make it so that the figure and the axes labels are not cut off
    plt.tight_layout()

    # Save the figure
    fig.savefig(
        "images/plots/three_worlds_example_strategies.png", dpi=300, bbox_inches="tight"
    )

    plt.show()
