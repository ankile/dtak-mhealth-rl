import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from src.utils.enums import TransitionMode

from src.utils.transition_matrix import make_absorbing
from src.visualization.worldviz import plot_world_reward
from src.worlds.mdp2d import Experiment_2D


def cliff_reward(
    x,
    y,
    s,
    d,
    latent_reward=0,
    allow_disengage=False,
    disengage_reward=0,
) -> dict:
    """
    Creates a cliff on the bottom of the gridworld.
    The start state is the bottom left corner.
    The goal state is the bottom right corner.
    Every cell in the bottom row except the goal and start state is a cliff.

    The agent needs to walk around the cliff to reach the goal.

    :param x: width of the gridworld
    :param y: height of the gridworld
    :param d: reward for reaching the goal
    :param c: latent cost
    :param T: the transition matrix
    :param s: cost of falling off the cliff
    :param allow_disengage: whether to allow the agent to disengage in the world

    returns a dictionary of rewards for each state in the gridworld.
    """
    # Create the reward dictionary
    reward_dict = {}
    for i in range(x * y):
        reward_dict[i] = latent_reward  # add latent cost

    # Define the world boundaries
    cliff_begin_x = 1
    cliff_end_x = x - 1
    cliff_y = y - (1 + int(allow_disengage))

    # Set the goal state
    reward_dict[x * cliff_y + x - 1] = d

    # Set the cliff states
    for i in range(cliff_begin_x, cliff_end_x):
        reward_dict[x * cliff_y + i] = s

    # Set the disengage states
    if allow_disengage:
        for i in range(0, x):
            reward_dict[x * (y - 1) + i] = disengage_reward

    return reward_dict


def make_cliff_transition(
    T,
    height,
    width,
    allow_disengage=False,
    **kwargs,
) -> np.ndarray:
    """
    Makes the cliff absorbing.
    """

    cliff_begin_x = 1
    cliff_end_x = width - 1
    # The cliff is one cell above the bottom row when we allow for disengagement
    cliff_y = height - (1 + int(allow_disengage))

    # Make the cliff absorbing
    T_new = T.copy()

    for i in range(cliff_begin_x, cliff_end_x):
        idx = width * cliff_y + i
        make_absorbing(T_new, idx)

    if allow_disengage:
        for i in range(0, width):
            idx = width * (height - 1) + i
            make_absorbing(T_new, idx)

    return T_new


def make_cliff_experiment(
    prob,
    gamma,
    height,
    width,
    reward_mag,
    small_r_mag=0,
    neg_mag=-1e8,
    latent_reward=0,
    disengage_reward=0,
    allow_disengage=False,
) -> Experiment_2D:
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

    # Adding the smaller reward
    cliff_dict[width - 1] = small_r_mag

    experiment = Experiment_2D(
        height,
        width,
        action_success_prob=prob,
        rewards_dict=cliff_dict,
        gamma=gamma,
        transition_mode=TransitionMode.FULL,
    )

    T_new = make_cliff_transition(
        T=experiment.mdp.T,
        height=height,
        width=width,
        allow_disengage=allow_disengage,
    )

    experiment.mdp.T = T_new

    return experiment


if __name__ == "__main__":
    params = {
        "prob": 0.74,
        "gamma": 0.96,
        "height": 4,
        "width": 8,
        "reward_mag": 5e2,
        "small_r_mag": 200,  # small_mag of 0 = normal cliff world
        "neg_mag": -1e8,
        "latent_reward": -1,
        "disengage_reward": None,
        "allow_disengage": False,
    }

    experiment = make_cliff_experiment(**params)

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
        save_heatmap=False,
        show_heatmap=False,
        heatmap_ax=ax3,
        heatmap_mask=mask,
        base_dir="local_images",
        label_precision=1,
    )

    # set titles for subplots
    ax1.set_title("Parameters", fontsize=16)
    ax2.set_title("World Rewards", fontsize=16)
    ax3.set_title("Optimal Policy for Parameters", fontsize=16)

    # Show the plot
    plt.show()
