import itertools
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker
from tqdm import tqdm


# Import the cliff world
from cliff_world import cliff_experiment
from worlds.mdp2d import MDP_2D


def follow_policy(policy, height, width, initial_state, terminal_states):
    action_dict = {0: "L", 1: "R", 2: "U", 3: "D"}
    state = initial_state
    actions_taken = []

    while state not in terminal_states:
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


if __name__ == "__main__":
    # default_params = {
    #     "prob": 0.72,
    #     "gamma": 0.89,
    #     "height": 3,
    #     "width": 8,
    #     "reward_mag": 1e2,
    #     "neg_mag": -1e2,
    #     "latent_reward": 0,
    #     "disengage_reward": None,
    #     "allow_disengage": False,
    # }

    # experiment = cliff_experiment(
    #     setup_name="Cliff",
    #     **default_params,
    # )

    params = {
        "height": 4,
        "width": 8,
        "reward_mag": 1e2,
        "neg_mag": -1e2,
        "latent_reward": 0,
        "disengage_reward": None,
        "allow_disengage": False,
    }

    # h, w = default_params["height"], default_params["width"]
    h, w = params["height"], params["width"]

    gammas = np.linspace(0.5, 0.999, 30)
    probs = np.linspace(0.5, 0.999, 30)

    data = np.zeros((len(probs), len(gammas)), dtype=int)
    policies = {}
    p2idx = {}

    # Make plot with 5 columns where the first column is the parameters
    # and the two plots span two columns each

    # create figure with 5 columns
    fig, ax = plt.subplots(figsize=(6, 4))

    # Adjust layout and spacing (make room for titles)
    plt.subplots_adjust(top=0.9)

    # Set the starting state to be the bottom left corner
    starting_state = (h - 1) * w

    pbar = tqdm(total=len(probs) * len(gammas))

    for (i, j), (prob, gamma) in zip(
        itertools.product(range(len(probs)), range(len(gammas))),
        itertools.product(probs, gammas),
    ):
        pbar.set_description(f"Prob: {prob:>3.2f}, Gamma: {gamma:>3.2f}")

        experiment = cliff_experiment(
            setup_name="Cliff",
            **{**params, "prob": prob, "gamma": gamma},
        )

        experiment.mdp.solve(
            setup_name="Cliff",
            policy_name="Baseline World",
            save_heatmap=False,
            show_heatmap=False,
            # TODO: Add in ax3 again
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
        pbar.update(1)

    # parameter_text = ", ".join([f"{k}: {v}" for k, v in default_params.items()])
    parameter_text = ", ".join([f"{k}: {v}" for k, v in params.items()])
    parameter_text = (
        parameter_text[: len(parameter_text) // 2]
        + "\n"
        + parameter_text[len(parameter_text) // 2 :]
    )

    # set the number of tick labels to display
    num_ticks = 10

    # compute the indices to use for the tick labels
    gamma_indices = np.round(np.linspace(0, len(gammas) - 1, num_ticks)).astype(int)
    prob_indices = np.round(np.linspace(0, len(probs) - 1, num_ticks)).astype(int)

    # create the tick labels
    gamma_ticks = [round(gammas[i], 2) for i in gamma_indices]
    prob_ticks = [round(probs[i], 2) for i in prob_indices]

    # plot the heatmap
    ax = sns.heatmap(data, annot=False, cmap="Blues", fmt="d", ax=ax, cbar=False)

    # set the tick labels and positions
    ax.xaxis.set_major_locator(ticker.FixedLocator(gamma_indices))
    ax.set_xticklabels(gamma_ticks, rotation=90, size=8)
    ax.yaxis.set_major_locator(ticker.FixedLocator(prob_indices))
    ax.set_yticklabels(prob_ticks, size=8, rotation=0)

    # invert the y-axis
    ax.invert_yaxis()

    ax.set_xlabel("Gamma")
    ax.set_ylabel("Confidence")

    fig.suptitle(parameter_text, fontsize=8)
    plt.tight_layout()

    plt.show()
