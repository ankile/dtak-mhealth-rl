import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from src.utils.enums import TransitionMode

from src.visualization.worldviz import plot_world_reward
from src.worlds.mdp2d import Experiment_2D


def get_goal_states(h, w) -> set:
    """
    Returns a list of goal states for a gridworld of size (h, w).
    """
    goal_states = {0, w - 1}

    return goal_states


def riverswim_reward(
    width,
    big_r,
    small_r,
    **kwargs,
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
    for i in range(width):
        reward_dict[i] = 0  # add latent cost

    # set rewards/goal states
    reward_dict[0] = small_r
    reward_dict[width - 1] = big_r

    return reward_dict


def make_riverswim_transition(width, prob, **kwargs) -> np.ndarray:
    """
    Sets up the transition matrix for the riverswim environment.
    """
    T_new = np.zeros(
        (4, width, width)
    )  # reset transition matrix, which also removes absorbing states

    # set left behavior (0): deterministic
    T_new[0, 0, 0] = 1
    for row in range(1, width):
        T_new[0, row, row - 1] = 1

    # set right behavior (1): based on the confidence
    # set first row
    T_new[1, 0, 0] = 1 - prob
    T_new[1, 0, 1] = prob

    left_stay_ratio = 1 / 13  # can vary this ratio, but for now it's stuck at 1/13

    for row in range(1, width - 1):  # set the rows in between
        T_new[1, row, row + 1] = prob  # intended action prob = confidence
        T_new[1, row, row - 1] = left_stay_ratio * (
            1 - prob
        )  # left action prob = ratio * (1-confidence)
        T_new[1, row, row] = (1 - left_stay_ratio) * (
            1 - prob
        )  # right action prob = 1 - intended - left

    # set last row
    T_new[1, width - 1, width - 2] = 1 - prob
    T_new[1, width - 1, width - 1] = prob

    # set up and down behavior (2, 3): deterministic
    for row in range(width):
        T_new[2, row, row] = 1
        T_new[3, row, row] = 1
    return T_new


def make_riverswim_experiment(
    prob,
    height,
    width,
    big_r,
    small_r,
    **kwargs,
) -> Experiment_2D:
    riverswim_dict = riverswim_reward(
        width=width,
        big_r=big_r,
        small_r=small_r,
    )

    experiment = Experiment_2D(
        height=height,
        width=width,
        rewards_dict=riverswim_dict,
        transition_mode=TransitionMode.FULL,
    )

    T_new = make_riverswim_transition(
        T=experiment.mdp.T,
        height=1,
        width=width,
        prob=prob,
    )

    experiment.mdp.T = T_new

    return experiment


if __name__ == "__main__":
    params = {
        "prob": 0.72,
        "gamma": 0.99,
        "height": 1,
        "width": 5,
        "big_r": 5,
        "small_r": 1,
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

    plot_world_reward(experiment, setup_name="Riverswim", ax=ax2, show=False, mask=mask)

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
