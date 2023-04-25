from datetime import datetime
import itertools
import os
import pickle
from typing import Callable, Tuple
from matplotlib import pyplot as plt
from functools import partial

import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


# Import the cliff world
from cliff_world import cliff_experiment
from utils.cliff import cliff_transition
from worlds.mdp2d import Experiment_2D
from utils.policy import follow_policy, get_all_absorbing_states, param_generator
from visualization.strategy import make_cliff_strategy_heatmap


def run_experiment(
    experiment: Experiment_2D,
    params: dict,
    gammas: np.ndarray,
    probs: np.ndarray,
):
    data = np.zeros((len(probs), len(gammas)), dtype=int)
    policies = {}
    p2idx = {}

    terminal = get_all_absorbing_states(experiment.mdp)

    for (i, prob), (j, gamma) in itertools.product(enumerate(probs), enumerate(gammas)):
        experiment.set_user_params(
            prob=prob, gamma=gamma, transition_func=cliff_transition
        )

        experiment.mdp.solve(
            save_heatmap=False,
            show_heatmap=False,
            heatmap_ax=None,
            heatmap_mask=None,
            label_precision=1,
        )

        policy_str = follow_policy(
            experiment.mdp.policy,
            height=params["height"],
            width=params["width"],
            initial_state=(params["height"] - 1) * params["width"],
            terminal_states=terminal,
        )

        policies[(prob, gamma)] = policy_str
        if policy_str not in p2idx:
            p2idx[policy_str] = len(p2idx)

        data[i, j] = p2idx[policy_str]

    return data, p2idx


def run_one_world(
    param_value: Tuple[str, float],
    *,
    create_experiment_func: Callable[..., Experiment_2D],
    default_params: dict,
    probs: np.ndarray,
    gammas: np.ndarray,
):
    param, value = param_value
    params = {**default_params, param: value}
    experiment: Experiment_2D = create_experiment_func(**params)
    data, p2idx = run_experiment(
        experiment,
        params,
        gammas,
        probs,
    )
    return data, p2idx, param, value


def init_outdir(setup_name):
    setup_name = setup_name.replace(" ", "_").lower()
    output_dir = f"local_images/{setup_name}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def run_param_sweep(
    output_dir,
    run_parallel,
    default_params,
    cols,
    granularity,
    probs,
    gammas,
    search_parameters,
    rows,
):
    run_one_world_partial = partial(
        run_one_world,
        create_experiment_func=cliff_experiment,
        default_params=default_params,
        probs=probs,
        gammas=gammas,
    )

    n_processes = os.cpu_count() if run_parallel else 1
    print(f"Running {n_processes} processes...")
    start = datetime.now()
    with Pool(processes=n_processes) as pool:
        strategy_data = list(
            tqdm(
                pool.imap(run_one_world_partial, param_generator(search_parameters)),
                total=rows * cols,
                desc=f"Running with cols={cols}, rows={rows}, granularity={granularity}",
                ncols=0,
            )
        )
    print(f"Finished in {datetime.now() - start}")

    partial_strategy_heatmap = partial(
        make_cliff_strategy_heatmap,
        gammas=gammas,
        probs=probs,
        annot=False,
        ax_labels=False,
        num_ticks=10,
    )

    # Create the figure and axes to plot on
    fig, axs = plt.subplots(
        nrows=rows, ncols=cols, figsize=(16, 3 * rows), sharex=True, sharey=True
    )
    fig.subplots_adjust(top=0.9)

    for (data, p2idx, param, value), ax in zip(strategy_data, axs.flatten()):
        partial_strategy_heatmap(
            results=data,
            p2idx=p2idx,
            title=f"{param}={value:.2f}",
            ax=ax,
        )

    # Shoow the full plot at the end
    fig.suptitle(
        "Cliff World Parameter Search:\n"
        + ", ".join(f"{k}={v}" for k, v in default_params.items())
    )
    plt.tight_layout()

    # Set the x and y labels
    for ax in axs[-1]:
        ax.set_xlabel("Gamma")
    for ax in axs[:, 0]:
        ax.set_ylabel("Prob")

    # Save the figure
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fig.savefig(f"{output_dir}/cliff_param_search_{now}.png", dpi=300)

    # Add metadata to the heatmap info
    print("Saving metadata with length", len(strategy_data))
    pickle_data = {
        "strategy_data": strategy_data,
        "grid_dimensions": (rows, cols),
        "search_parameters": search_parameters,
        "default_params": default_params,
        "probs": probs,
        "gammas": gammas,
    }

    # Save the heatmap info to file
    with open(f"{output_dir}/cliff_param_search_{now}.pkl", "wb") as f:
        pickle.dump(pickle_data, f)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # === Start of setup ===
    output_dir = init_outdir(setup_name="Cliff World Param Viz")

    run_parallel = True

    default_params = {
        "height": 5,
        "width": 9,
        "reward_mag": 1e2,
        "small_r_mag": 0,
        "neg_mag": -1e8,
        "latent_reward": 0,
        # These are off by default because they turn the world into a compound world
        "disengage_reward": 0,
        "allow_disengage": False,
    }

    # Set the number of subplots per row
    cols = 8  # 5, 7, 9

    # Set the number of scales and gammas to use
    granularity = 20  # 5, 10, 20

    # Set up parameters to search over
    probs = np.linspace(0.3, 0.99, granularity)
    gammas = np.linspace(0.4, 0.99, granularity)

    search_parameters = {
        "width": np.arange(7 - int(cols / 2), 7 + round(cols / 2)) + 1,
        "height": np.arange(7 - int(cols / 2), 7 + round(cols / 2)) + 1,
        # The rest of these are not interesting to change for different reasons
        # "reward_mag": np.linspace(100, 500, cols),
        # "small_r_mag": np.linspace(25, 100, cols),
        # "neg_mag": np.linspace(-20, 0, cols),
        # "latent_reward": np.linspace(-3, 0, cols),
        # "disengage_reward": np.linspace(0, 10, cols),
        # "prob": np.linspace(0.5, 0.95, cols),  # Don't search over prob
    }

    rows = len(search_parameters)

    # === End of setup ===

    run_param_sweep(
        output_dir,
        run_parallel,
        default_params,
        cols,
        granularity,
        probs,
        gammas,
        search_parameters,
        rows,
    )
