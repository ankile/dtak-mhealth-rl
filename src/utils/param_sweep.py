from collections import namedtuple
import itertools
import os
import pickle
from datetime import datetime
from functools import partial

from typing import Callable, Dict, Tuple, cast
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from src.utils.multithreading import OptionalPool

from src.utils.policy import follow_policy, get_all_absorbing_states, param_generator
from src.visualization.strategy import make_general_strategy_heatmap
from src.worlds.mdp2d import Experiment_2D

ExperimentResult = namedtuple("ExperimentResult", ["data", "p2idx", "probs"])


def run_experiment(
    experiment: Experiment_2D,
    transition_matrix_func: Callable,
    params: dict,
    gammas: np.ndarray,
    probs: np.ndarray,
    start_state: int,
    realized_probs_indices: list | None = None,
    goal_states: set | None = None,
) -> ExperimentResult:
    """
    Run an experiment with a given set of parameters and return the results.

    The results are:
    - data: a matrix of shape (len(probs), len(gammas)) where each entry is an index
            into the list of policies
    - p2idx: a dictionary mapping policies to indices
    - realized_probs: a matrix of shape (len(probs), len(gammas)) where each entry is
                        the realized probability of transitioning to the next state,
                        i.e., the probability of the transition that actually occurs
                        under the given definition of underconfidence (underconfidence/pessimism)

    If vanilla underconfidence or pessimism is used is implicitly determined by whether
    `realized_probs_indices` is None or a function that determines what entry in `T` to
    use as the realized probability.
    """

    data = np.zeros((len(probs), len(gammas)), dtype=np.int32)
    policies = {}
    p2idx: Dict[str, int] = {}

    realized_probs = np.zeros(probs.shape)
    if goal_states is None:
        goal_states = get_all_absorbing_states(experiment.mdp)

    for (i, prob), (j, gamma) in itertools.product(enumerate(probs), enumerate(gammas)):
        experiment.set_user_params(
            prob=prob,
            gamma=gamma,
            transition_func=transition_matrix_func,
            use_pessimistic=realized_probs_indices is not None,
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
            initial_state=start_state,
            goal_states=goal_states,
        )

        policies[(prob, gamma)] = policy_str
        if policy_str not in p2idx:
            p2idx[policy_str] = len(p2idx)

        data[i, j] = p2idx[policy_str]

        if realized_probs_indices is not None:
            a, s1, s2 = realized_probs_indices
            realized_probs[i] = experiment.mdp.T[a, s1, s2]
        else:
            realized_probs[i] = prob

    return ExperimentResult(data, p2idx, realized_probs)


def run_one_world(
    param_value: Tuple[str, float],
    *,
    create_experiment_func: Callable[..., Experiment_2D],
    transition_matrix_func: Callable[..., np.ndarray],
    default_params: dict,
    probs: np.ndarray,
    gammas: np.ndarray,
    get_start_state: Callable[[int, int], int],
    get_realized_probs_indices: Callable[[int, int], list] | None = None,
    get_goal_states: Callable[[int, int], set] | None = None,
):
    param, value = param_value
    params = {**default_params, param: value}
    experiment: Experiment_2D = create_experiment_func(**params)
    h, w = params["height"], params["width"]

    realized_probs_indices = (
        get_realized_probs_indices(h, w)
        if get_realized_probs_indices is not None
        else None
    )

    goal_states = get_goal_states(h, w) if get_goal_states is not None else None

    data, p2idx, realized_probs = run_experiment(
        experiment,
        transition_matrix_func,
        params,
        gammas,
        probs,
        start_state=get_start_state(h, w),
        realized_probs_indices=realized_probs_indices,
        goal_states=goal_states,
    )
    return data, p2idx, param, value, realized_probs


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
    get_start_state,
    granularity,
    gammas,
    probs=None,
    scalers=None,  # When this is defined, that means we are using the scaled version, i.e., pessimism
    get_realized_probs_indices: Callable | None = None,
    get_goal_states: Callable | None = None,
    run_parallel=False,
):
    assert not (
        probs is not None and scalers is not None
    ), "Cannot have both probs and scalers defined"

    assert (
        probs is not None or scalers is not None
    ), "Must have either probs or scalers defined"

    assert (
        get_realized_probs_indices is not None or scalers is None
    ), "Must have realized_prob_coord when scalers is defined"

    if probs is None:
        probs = cast(np.ndarray, scalers)

    run_one_world_partial = partial(
        run_one_world,
        create_experiment_func=create_experiment_func,
        transition_matrix_func=transition_matrix_func,
        default_params=default_params,
        probs=probs,
        gammas=gammas,
        get_start_state=get_start_state,
        get_realized_probs_indices=get_realized_probs_indices,
        get_goal_states=get_goal_states,
    )

    n_processes = (os.cpu_count() if run_parallel else 1) or 1
    print(f"Running {n_processes} processes...")
    start = datetime.now()
    with OptionalPool(processes=n_processes) as pool:
        strategy_data = list(
            tqdm(
                pool.imap(run_one_world_partial, param_generator(search_parameters)),
                total=rows * cols,
                desc=f"Running with cols={cols}, rows={rows}, granularity={granularity}",
                ncols=0,
            )
        )
    print(f"Finished in {datetime.now() - start}")

    # Create the figure and axes to plot on
    fig, axs = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=(round(2.5 * cols), round(2.5 * rows)),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(top=0.9)

    for (data, p2idx, param, value, realized_probs), ax in zip(
        strategy_data, axs.flatten()
    ):
        make_general_strategy_heatmap(
            results=data,
            probs=realized_probs,
            p2idx=p2idx,
            title=f"{param}={value:.2f}",
            ax=ax,
            gammas=gammas,
            annot=False,
            ax_labels=False,
            num_ticks=10,
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
