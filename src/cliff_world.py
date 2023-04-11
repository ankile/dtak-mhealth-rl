import os
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from worlds.mdp2d import Experiment_2D

# Import the cliff world
from utils.cliff import cliff_reward, cliff_transition
from visualization.worldviz import plot_world_reward


def cliff_experiment(
    setup_name: str,
    height,
    width,
    prob,
    gamma,
    reward_mag,
    neg_mag=-np.inf,
    latent_reward=0,
    disengage_reward=0,
    allow_disengage=False,
) -> Experiment_2D:
    if not os.path.exists(savepath := f"local_images/{setup_name}"):
        os.makedirs(savepath)

    # Add one row for the disengage state if allowed
    if allow_disengage:
        height += 1

    cliff_dict = cliff_reward(
        x=width,
        y=height,
        s=neg_mag,
        d=reward_mag,
        latent_reward=latent_reward,
        disengage_reward=disengage_reward,
        allow_disengage=allow_disengage,
    )

    experiment = Experiment_2D(
        height,
        width,
        rewards_dict=cliff_dict,
        gamma=gamma,
        action_success_prob=prob,
        transition_mode="full",
    )

    T_new = cliff_transition(
        T=experiment.mdp.T,
        x=width,
        y=height,
        allow_disengage=allow_disengage,
    )

    experiment.mdp.T = T_new

    return experiment


def get_all_absorbing_states(T, height, width):
    absorbing_states = []

    for state in range(height * width):
        for action in range(4):
            if T[action, state, state] == 1:
                absorbing_states.append(state)

    return absorbing_states


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


if __name__ == "__main__":
    params = {
        "prob": 0.72,
        "gamma": 0.89,
        "height": 3,
        "width": 8,
        "reward_mag": 1e2,
        "neg_mag": -1e2,
        "latent_reward": 0,
        "disengage_reward": None,
        "allow_disengage": False,
    }

    experiment = cliff_experiment(
        setup_name="Cliff",
        **params,
    )

    # Make plot with 5 columns where the first column is the parameters
    # and the two plots span two columns each

    # create figure with 5 columns
    fig = plt.figure(figsize=(12, 4))
    gs = GridSpec(1, 5, figure=fig)

    # add text to first column
    ax1 = fig.add_subplot(gs[0, 0])  # type: ignore
    ax1.axis("off")

    # add subplots to remaining 4 columns
    ax2 = fig.add_subplot(gs[0, 1:3])  # type: ignore
    ax3 = fig.add_subplot(gs[0, 3:5])  # type: ignore

    # Adjust layout and spacing (make room for titles)
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Add the parameters to the first subplot
    ax1.text(
        0.05,
        0.95,
        "\n".join([f"{k}: {v}" for k, v in params.items()]),
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax1.transAxes,
    )

    # Create a mask for the bottom row if the user is allowed to disengage
    mask = None
    if params["allow_disengage"]:
        mask = np.zeros(
            (params["height"] + int(params["allow_disengage"]), params["width"])
        )
        mask[-1, :] = 1

    plot_world_reward(experiment, setup_name="Cliff", ax=ax2, show=False, mask=mask)

    experiment.mdp.solve(
        setup_name="Cliff",
        policy_name="Baseline World",
        save_heatmap=False,
        show_heatmap=False,
        # TODO: Add in ax3 again
        heatmap_ax=None,
        heatmap_mask=mask,
        base_dir="local_images",
        label_precision=1,
    )

    # TODO: Debugging purposes
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

    print(policy_str)

    # set titles for subplots
    ax1.set_title("Parameters", fontsize=16)
    ax2.set_title("World Rewards", fontsize=16)
    ax3.set_title("Optimal Policy for Parameters", fontsize=16)

    # Show the plot
    plt.show()
