import itertools
import os
import pickle
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from typing import Callable, Tuple

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.policy import follow_policy, get_all_absorbing_states, param_generator
from visualization.strategy import make_general_strategy_heatmap
from worlds.mdp2d import Experiment_2D


def run_experiment(
    experiment: Experiment_2D,
    transition_matrix_func: Callable,
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
            prob=prob, gamma=gamma, transition_func=transition_matrix_func
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
    transition_matrix_func: Callable[..., np.ndarray],
    default_params: dict,
    probs: np.ndarray,
    gammas: np.ndarray,
):
    param, value = param_value
    params = {**default_params, param: value}
    experiment: Experiment_2D = create_experiment_func(**params)
    data, p2idx = run_experiment(
        experiment,
        transition_matrix_func,
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
    *,
    setup_name,
    default_params,
    search_parameters,
    create_experiment_func,
    transition_matrix_func,
    rows,
    cols,
    granularity,
    probs,
    gammas,
    run_parallel,
):
    run_one_world_partial = partial(
        run_one_world,
        create_experiment_func=create_experiment_func,
        transition_matrix_func=transition_matrix_func,
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
        make_general_strategy_heatmap,
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
        f"{setup_name}:\n" + ", ".join(f"{k}={v}" for k, v in default_params.items())
    )
    plt.tight_layout()

    # Set the x and y labels
    for ax in axs[-1]:
        ax.set_xlabel("Gamma")
    for ax in axs[:, 0]:
        ax.set_ylabel("Prob")

    # Determine the output directory
    output_dir = init_outdir(setup_name)

    # Save the figure
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fig.savefig(f"{output_dir}/{now}_vizualization.png", dpi=300)

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
    with open(f"{output_dir}/{now}_metadata.pkl", "wb") as f:
        pickle.dump(pickle_data, f)

    # Show the plot
    plt.show()
