import itertools
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
from tqdm import tqdm


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


def param_generator(parameters, axs):
    axs = axs.reshape(-1, axs.shape[-1])
    for i, (param_name, param_values) in enumerate(parameters.items()):
        ax_row = axs[i]
        for value, ax in zip(param_values, ax_row):
            yield param_name, value, ax


if __name__ == "__main__":
    default_params = {
        "height": 4,
        "width": 8,
        "reward_mag": 1e2,
        "neg_mag": -1e2,
        "latent_reward": 0,
        "disengage_reward": None,
        "allow_disengage": False,
    }

    # Set the number of subplots per row
    cols = 9  # 5, 7, 9

    # Set the number of scales and gammas to use
    granularity = 10  # 5, 10, 20

    # Set up parameters to search over
    probs = np.linspace(0.3, 0.99, granularity)
    gammas = np.linspace(0.4, 0.99, granularity)

    parameters = {
        "reward_mag": np.linspace(100, 500, cols),
        "neg_mag": np.linspace(-20, 0, cols),
        "latent_reward": np.linspace(-3, 0, cols),
        "prob": np.linspace(0.5, 0.95, cols),
        "width": list(range(6 - int(cols / 2), 6 + int(cols / 2) + 1)),
        "height": list(range(6 - int(cols / 2), 6 + int(cols / 2) + 1)),
    }

    rows = len(parameters)

    # Create the figure and axes to plot on
    fig, axs = plt.subplots(
        nrows=rows, ncols=cols, figsize=(16, 9), sharex=True, sharey=True
    )
    fig.subplots_adjust(top=0.9)

    pbar = tqdm(total=rows * cols)
    for param, value, ax in param_generator(parameters, axs):
        pbar.set_description(f"Running {param}={value:.2f}")
        config = {**default_params, param: value}
        data, p2idx = run_cliff_experiment(
            config,
            gammas,
            probs,
        )

        make_cliff_strategy_heatmap(
            data,
            gammas,
            probs,
            ax=ax,
            p2idx=p2idx,
            title=f"{param}={value:.2f}",
            annot=False,
            ax_labels=False,
            num_ticks=10,
        )

        pbar.update(1)

    # Shoow the full plot at the end
    fig.suptitle(
        "Cliff World Parameter Search:\n"
        + ", ".join(f"{k}={v}" for k, v in default_params.items())
    )
    plt.tight_layout()
    plt.show()
