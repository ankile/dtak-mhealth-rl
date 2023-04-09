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
    absorbing_states = []

    for state in range(height * width):
        for action in range(4):
            if T[action, state, state] == 1:
                absorbing_states.append(state)

    return absorbing_states


if __name__ == "__main__":
    default_params = {
        "prob": 0.7,
        "gamma": 0.90,
        "height": 4,
        "width": 8,
        "reward_mag": 1e3,
        "neg_mag": -1e4,
        "latent_reward": 0,
        "disengage_reward": 1e1,
        "allow_disengage": False,
    }

    h, w = default_params["height"], default_params["width"]

    gammas = np.linspace(0.2, 0.999, 10)
    probs = np.linspace(0.5, 0.99, 10)

    data = np.zeros((len(probs), len(gammas)), dtype=int)
    policies = {}
    p2idx = {}

    # Make plot with 5 columns where the first column is the parameters
    # and the two plots span two columns each

    # create figure with 5 columns
    fig, ax = plt.subplots(figsize=(8, 4))

    # Adjust layout and spacing (make room for titles)
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)

    experiment = cliff_experiment(
        setup_name="Cliff",
        **default_params,
    )

    terminal_states = get_all_absorbing_states(experiment.mdp.T, h, w)
    # Set the starting state to be the bottom left corner
    starting_state = (h - 1) * w

    pbar = tqdm(total=len(probs) * len(gammas))

    for (i, j), (prob, gamma) in zip(
        itertools.product(range(len(probs)), range(len(gammas))),
        itertools.product(probs, gammas),
    ):
        experiment.mdp.reset()

        S, A, T, R, gamma = experiment.make_MDP_params(
            action_success_prob=prob,
            transition_mode="full",
            gamma=gamma,
            height=h,
            width=w,
            rewards_dict=experiment.rewards_dict,
        )

        experiment.mdp = MDP_2D(S, A, T, R, gamma)

        values, policy = experiment.mdp.solve(
            setup_name="Cliff",
            policy_name="Baseline World",
            save_heatmap=False,
            show_heatmap=False,
            heatmap_ax=None,
            heatmap_mask=None,
            base_dir="local_images",
            label_precision=1,
        )
        policy_str = follow_policy(
            policy,
            default_params["height"],
            default_params["width"],
            starting_state,
            terminal_states,
        )

        policies[(prob, gamma)] = policy_str
        if policy_str not in p2idx:
            p2idx[policy_str] = len(p2idx)

        data[i, j] = p2idx[policy_str]
        pbar.update(1)

    print(p2idx)
    print(policies)

    parameter_text = ", ".join([f"{k}: {v}" for k, v in default_params.items()])
    ax.set_title(parameter_text, fontsize=8)

    # set the number of tick labels to display
    num_ticks = 10

    # compute the indices to use for the tick labels
    gamma_indices = np.round(np.linspace(0, len(gammas) - 1, num_ticks)).astype(int)
    prob_indices = np.round(np.linspace(0, len(probs) - 1, num_ticks)).astype(int)

    # create the tick labels
    gamma_ticks = [round(gammas[i], 2) for i in gamma_indices]
    prob_ticks = [round(probs[i], 2) for i in prob_indices]

    # plot the heatmap
    ax = sns.heatmap(data, annot=True, cmap="Blues", fmt="d", ax=ax, cbar=False)

    # set the tick labels and positions
    ax.xaxis.set_major_locator(ticker.FixedLocator(gamma_indices))
    ax.set_xticklabels(gamma_ticks, rotation=90, size=8)
    ax.yaxis.set_major_locator(ticker.FixedLocator(prob_indices))
    ax.set_yticklabels(prob_ticks, size=8, rotation=0)

    # invert the y-axis
    ax.invert_yaxis()

    ax.set_xlabel("Gamma")
    ax.set_ylabel("Confidence")

    plt.show()
