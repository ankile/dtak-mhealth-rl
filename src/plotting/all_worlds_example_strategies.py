# Standard imports
from functools import partial
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.plotting.config import (
    FIG_AXIS_FONT_SIZE,
    FIG_HALF_SIZE,
    FIG_LEGEND_FONT_SIZE,
    FIG_TITLE_FONT_SIZE,
)

# Import function from `param_sweep.py` to run one experiment
from src.utils.param_sweep import run_experiment
from src.utils.small_big import make_smallbig_experiment
from src.utils.transition_matrix import id_func
from src.visualization.strategy import make_general_strategy_heatmap

# Import config functions for the wall experiment
import src.param_sweeps.wall_world as wall_world

# Import config functions for the cliff experiment
import src.param_sweeps.cliff_world as cliff_world
from src.utils.cliff import make_cliff_experiment, make_cliff_transition

# Import config functions for the smallbig experiment
import src.param_sweeps.smallbig_world as smallbig_world
from src.utils.wall import make_wall_experiment

# Import config functions for the Chain experiment
import src.param_sweeps.chain_world as chain_world
from src.utils.chain import make_chain_experiment, make_chain_transition

# Import config functions for the Riwerswim experiment
import src.param_sweeps.riverswim_world as riverswim_world
from src.utils.riverswim import make_riverswim_experiment, make_riverswim_transition

# Import config functions for the gamblers experiment
import src.param_sweeps.gamblers_world as gamblers_world
from src.utils.gamblers import make_gamblers_experiment, make_gamblers_transition


def run_wall(experiment, gammas: np.ndarray, probs: np.ndarray):
    h, w = experiment.height, experiment.width

    return run_experiment(
        experiment,
        transition_matrix_func=id_func,
        params=wall_world.default_params,
        gammas=gammas,
        probs=probs,
        start_state=wall_world.get_start_state(h, w),
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


def run_riverswim(experiment, gammas: np.ndarray, probs: np.ndarray):
    h, w = experiment.height, experiment.width

    return run_experiment(
        experiment,
        transition_matrix_func=make_riverswim_transition,
        params=riverswim_world.default_params,
        gammas=gammas,
        probs=probs,
        start_state=riverswim_world.get_start_state(h, w),
    )


def run_gamblers_cont(experiment, gammas: np.ndarray, probs: np.ndarray):
    h, w = experiment.height, experiment.width

    return run_experiment(
        experiment,
        transition_matrix_func=make_gamblers_transition,
        params=gamblers_world.default_params,
        gammas=gammas,
        probs=probs,
        start_state=gamblers_world.get_start_state(h, w),
    )


def run_gamblers_finish(experiment, gammas: np.ndarray, probs: np.ndarray):
    h, w = experiment.height, experiment.width

    return run_experiment(
        experiment,
        transition_matrix_func=partial(
            make_gamblers_transition, vary_continuation=False
        ),
        params=gamblers_world.default_params,
        gammas=gammas,
        probs=probs,
        start_state=gamblers_world.get_start_state(h, w),
    )


experiments = {
    "Wall": {
        "p2idx": {
            "Around": 0,
            "Through": 1,
        },
        "experiment": make_wall_experiment(
            height=2,
            width=5,
            reward_mag=250,
            neg_mag=-10,
            latent_cost=0,
        ),
        "run_func": run_wall,
    },
    "Cliff": {
        "p2idx": {
            "Hug cliff": 0,
            "Keep space": 1,
        },
        "experiment": make_cliff_experiment(
            height=5,
            width=10,
            reward_mag=100,
            small_r_mag=0,
        ),
        "run_func": run_cliff,
    },
    "SmallBig": {
        "p2idx": {
            "Small reward": 0,
            "Big reward": 1,
        },
        "experiment": make_smallbig_experiment(
            height=7,
            width=7,
            big_reward=100,
            small_reward_frac=0.2,
        ),
        "run_func": run_smallbig,
    },
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
            goal_mag=15,
            disengage_reward=-1,
            burden=-1,
        ),
        "run_func": run_chain,
    },
    "RiverSwim": {
        "p2idx": {
            "Upstream": 0,
            "Downstream": 1,
        },
        "default": 1,
        "experiment": make_riverswim_experiment(
            height=1,
            width=7,
            prob=0.8,
            gamma=0.9,
            big_r=5,
            small_r=1,
        ),
        "run_func": run_riverswim,
    },
    "Gambler's ($p_C$)": {
        "p2idx": {
            "Continue": 1,
            "Finish": 0,
        },
        "default": 1,
        "experiment": make_gamblers_experiment(
            width=5,
            prob=0.8,
            gamma=0.9,
            big_r=5,
            small_r=0,
            vary_continuation=True,
        ),
        "run_func": run_gamblers_cont,
    },
    "Gambler's ($p_F$)": {
        "p2idx": {
            "Continue": 1,
            "Finish": 0,
        },
        "default": 1,
        "experiment": make_gamblers_experiment(
            width=5,
            prob=0.8,
            gamma=0.9,
            big_r=5,
            small_r=0,
            vary_continuation=False,
        ),
        "run_func": run_gamblers_finish,
    },
}


if __name__ == "__main__":
    # Set up the parameters for the parameter sweep
    granularity = 20
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
            title=f"{name} Strategy Map",
            annot=False,
            ax_labels=True,
            num_ticks=3,
            title_fontsize=FIG_TITLE_FONT_SIZE,
            legend_fontsize=FIG_LEGEND_FONT_SIZE,
            tick_fontsize=FIG_AXIS_FONT_SIZE,
        )

        # Make it so that the figure and the axes labels are not cut off
        plt.tight_layout()

        # Save the figure
        fig.savefig(
            f"images/plots/strategy_map_{name}.pdf",
            dpi=300,
            bbox_inches="tight",
        )
