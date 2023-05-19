import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from enums import TransitionMode

from transition_matrix import make_absorbing
from src.visualization.worldviz import plot_world_reward
from src.worlds.mdp1d import Experiment_1D


def get_goal_states(h, w) -> set:
    """
    Returns a list of goal states for a gridworld of size (h, w).
    """
    goal_states = {0, w - 1}

    return goal_states


def riverswim_reward(
    length: int,
    big_r,
    small_r,
    latent_reward=0,
) -> dict:
    """
    Creates a cliff on the bottom of the gridworld.
    The start state is the bottom left corner.
    The goal state is the bottom right corner.
    Every cell in the bottom row except the goal and start state is a cliff.

    The agent needs to walk around the cliff to reach the goal.

    :param x: length of river swim
    :param R: big reward on right
    :param r: small reward on left
    :param latent_reward: latent cost
    :param T: the transition matrix
    :param allow_disengage: whether to allow the agent to disengage in the world (not applicable here)

    returns a dictionary of rewards for each state in the gridworld.
    """
    # Create the reward dictionary
    reward_dict = {}
    for i in range(length):
        reward_dict[i] = latent_reward  # add latent cost

    # set rewards/goal states
    reward_dict[0] = small_r
    reward_dict[length - 1] = big_r

    return reward_dict


def make_riverswim_transition(T, length, prob, allow_disengage=False) -> np.ndarray:
    """
    Sets up the transition matrix for the riverswim environment.
    """
    T_new = np.zeros((2, length, length))  # reset

    # left and right should be absorbing states automatically, but just in case
    make_absorbing(T_new, 0)
    make_absorbing(T_new, length - 1)

    # set left behavior (0): deterministic
    T_new[0, 0, 0] = T_new[-1, -1, -1] = 1
    for row in range(1, length - 1):
        T_new[0, row, row - 1] = 1

    # set right behavior (1): based on the confidence
    T_new[1, 0, 0] = T_new[-1, -1, -1] = 1
    for row in range(1, length - 1):
        T_new[
            1, row, row - 1
        ] = 0.05  # can change this based on confidence; not sure yet
        T_new[1, row, row] = 0.8 * prob
        T_new[1, row, row + 1] = (
            1 - 0.05 - 0.8 * prob
        )  # again, this formula is quite rough

    return T_new


def make_riverswim_experiment(
    length: int,
    prob,
    big_r,
    small_r,
    latent_reward=0,
) -> Experiment_1D:
    riverswim_dict = riverswim_reward(
        length=length,
        big_r=big_r,
        small_r=small_r,
        latent_reward=latent_reward,
    )

    experiment = Experiment_1D(
        height=1,
        width=length,
        rewards_dict=riverswim_dict,
        transition_mode=TransitionMode.FULL,
    )

    T_new = make_riverswim_transition(
        T=experiment.mdp.T,
        height=1,
        width=length,
        prob=prob,
    )

    experiment.mdp.T = T_new

    return experiment


if __name__ == "__main__":
    params = {
        "prob": 0.72,
        "gamma": 0.89,
        "length": 3,
        "big_r": 1,
        "small_r": 0.01,  # small_mag of 0 = normal cliff world
        "latent_reward": 0,
        "disengage_reward": None,
        "allow_disengage": False,
    }

    experiment = make_riverswim_experiment(**params)

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
