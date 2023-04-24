from datetime import datetime
import itertools
import os
import pickle
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from functools import partial

import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


# Import the cliff world
from cliff_world import cliff_experiment
from visualization.strategy import make_cliff_strategy_heatmap


def follow_policy(policy, height, width, initial_state, terminal_states):
    action_dict = {0: "L", 1: "R", 2: "U", 3: "D"}
    state = initial_state
    actions_taken = []
    seen_states = set()

    while state not in terminal_states and state not in seen_states:
        seen_states.add(state)
        row, col = state // width, state % width
        action = policy[row, col]
        actions_taken.append(action_dict[action])

        if action == 0:  # left
            col = max(0, col - 1)
        elif action == 1:  # right
            col = min(width - 1, col + 1)
        elif action == 2:  # up
            row = max(0, row - 1)
        elif action == 3:  # down
            row = min(height - 1, row + 1)

        state = row * width + col

    return "".join(actions_taken)


def get_all_absorbing_states(T, height, width):
    absorbing_states = set()

    for state in range(height * width):
        for action in range(4):
            if T[action, state, state] == 1:
                absorbing_states.add(state)
                break

    return absorbing_states


def run_cliff_experiment(
    params,
    gammas,
    probs,
):
    data = np.zeros((len(probs), len(gammas)), dtype=int)
    policies = {}
    p2idx = {}

    for (i, prob), (j, gamma) in itertools.product(enumerate(probs), enumerate(gammas)):
        experiment = cliff_experiment(
            setup_name="Cliff",
            **params,
            prob=prob,
            gamma=gamma,
        )

        experiment.mdp.solve(
            setup_name="Cliff",
            policy_name="Cliff Policy",
            save_heatmap=False,
            show_heatmap=False,
            heatmap_ax=None,
            heatmap_mask=None,
            base_dir="local_images",
            label_precision=1,
        )

        terminal = get_all_absorbing_states(
            experiment.mdp.T, params["height"], params["width"]
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


def param_generator(parameters):
    for i, (param_name, param_values) in enumerate(parameters.items()):
        for value in param_values:
            yield (param_name, value)


def run_one_world(default_params, probs, gammas, param_value):
    param, value = param_value
    config = {**default_params, param: value}
    data, p2idx = run_cliff_experiment(
        config,
        gammas,
        probs,
    )

    return data, p2idx, param, value


if __name__ == "__main__":
    # Naming the setup
    setup_name = "Cliff World Param Viz"
    setup_name = setup_name.replace(" ", "_").lower()
    output_dir = f"local_images/{setup_name}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    default_params = {
        "height": 5,
        "width": 9,
        "reward_mag": 1e2,
        "small_r_mag": 0,
        "neg_mag": -1e8,
        "latent_reward": 0,
        "disengage_reward": None,
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
        # "reward_mag": np.linspace(100, 500, cols),
        # "small_r_mag": np.linspace(25, 100, cols),
        # "neg_mag": np.linspace(-20, 0, cols),
        # "latent_reward": np.linspace(-3, 0, cols),
        "width": np.arange(7 - int(cols / 2), 7 + round(cols / 2)) + 1,
        "height": np.arange(7 - int(cols / 2), 7 + round(cols / 2)) + 1,
        # "disengage_reward": np.linspace(0, 10, cols),
        # "prob": np.linspace(0.5, 0.95, cols),  # Don't search over prob
    }

    rows = len(search_parameters)

    run_one_world_partial = partial(run_one_world, default_params, probs, gammas)

    n_processes = os.cpu_count()
    print(f"Running {n_processes} processes...")
    start = datetime.now()
    with Pool(processes=n_processes) as pool:
        strategy_data = list(
            tqdm(
                pool.imap(run_one_world_partial, param_generator(search_parameters)),
                total=rows * cols,
                desc=f"Running with cols={cols}, rows={rows}, granularity={granularity}",
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
