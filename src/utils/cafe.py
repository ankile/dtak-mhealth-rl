import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from src.utils.enums import TransitionMode

from src.visualization.worldviz import plot_world_reward
from src.worlds.mdp2d import Experiment_2D

HEIGHT, WIDTH = 13, 8


def c2s(i, j) -> int:
    """
    Coordinate to State.
    Converts a 2D coordinate to a 1D state.
    """
    return i * WIDTH + j


DONUT1 = c2s(10, 0)
DONUT2 = c2s(5, 2)
NOODLE = c2s(9, 6)
VEGETARIAN = c2s(0, 4)

"""
Schema for the 13 by 8 cafe world:

    0   1   2   3   4   5   6   7
0   .   .   .   .   V   .   .   .
1   .   .   .   .   |   .   .   .
2   .   .   .   +   +   -   -   +
3   .   .   .   |   .   .   .   |
4   .   .   .   |   .   .   .   |
5   .   .   D   +   .   .   .   |
6   .   .   .   |   .   .   .   |
7   .   .   .   |   .   .   .   |
8   .   .   .   |   -   -   +   +
9   .   .   .   |   .   .   N   .
10  D   -   -   +   .   .   .   .
11  .   .   .   |   .   .   .   .
12  .   .   .   S   .   .   .   .

V   = vegetarian
D   = donut
N   = noodle
S   = start
.   = wall
+-| = path

"""


def get_goal_states(*args) -> set:
    """
    Returns a set of goal states for the cafe world.
    """

    return set([DONUT1, DONUT2, NOODLE, VEGETARIAN])


def cafe_reward(
    vegetarian_reward,
    donut_reward,
    noodle_reward,
    **kwargs,
) -> dict:
    """
    Returns a dictionary of rewards for the cafe world.

    Parameters
    ----------
    vegetarian_reward: float
        Reward for the vegetarian option.
    donut_reward: float
        Reward for the donut option.
    noodle_reward: float
        Reward for the noodle option.

    Returns
    -------
    dict
        Dictionary of rewards.

    """

    return {
        VEGETARIAN: vegetarian_reward,
        DONUT1: donut_reward,
        DONUT2: donut_reward,
        NOODLE: noodle_reward,
    }


def make_cafe_transition(T, prob, **kwargs) -> np.ndarray:
    """
    Sets up the transition matrix for the cafe world.
    """
    T_new = T.copy()

    path = np.array(
        [
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 1, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 1, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ]
    )

    n_states = WIDTH * HEIGHT
    n_actions = 4  # [E, W, N, S]

    for state in range(n_states):
        for action in range(n_actions):
            if (
                path[state // WIDTH][state % WIDTH] == 0
            ):  # if the current state is off the path
                T_new[action][state] = np.zeros((n_states,))
                T_new[action][state][
                    state
                ] = 1  # the only possible transition is staying put
                continue

            # If, from the current state and action, there is a non-zero probability of transitioning to a state off the path
            # then the probability of transitioning to that state is 0

            non_zero = T_new[action][state] > 0
            off_path = path.flatten() == 0

            T_new[action][state][non_zero & off_path] = 0

            # normalize probabilities for the current action to sum to 1
            T_new[action][state] /= np.sum(T_new[action][state])

    return T_new


def make_cafe_experiment(
    prob=0.80,
    gamma=0.90,
    vegetarian_reward=200,
    donut_reward=50,
    noodle_reward=100,
    **kwargs,
) -> Experiment_2D:
    cafe_dict = cafe_reward(
        vegetarian_reward=vegetarian_reward,
        donut_reward=donut_reward,
        noodle_reward=noodle_reward,
    )

    experiment = Experiment_2D(
        height=13,
        width=8,
        gamma=gamma,
        rewards_dict=cafe_dict,
        transition_mode=TransitionMode.FULL,
    )

    T_new = make_cafe_transition(
        T=experiment.mdp.T,
        height=13,
        width=8,
        prob=prob,
    )

    experiment.mdp.T = T_new

    return experiment


if __name__ == "__main__":
    params = {
        "prob": 0.05,
        "gamma": 0.70,
        "vegetarian_reward": 2000,
        "donut_reward": 2000,
        "noodle_reward": 0,
    }

    experiment = make_cafe_experiment(**params)

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

    plot_world_reward(experiment, setup_name="Cafe", ax=ax2, show=False)

    experiment.mdp.solve(
        save_heatmap=False,
        show_heatmap=False,
        heatmap_ax=ax3,
        base_dir="local_images",
        label_precision=1,
    )

    # set titles for subplots
    ax1.set_title("Parameters", fontsize=16)
    ax2.set_title("World Rewards", fontsize=16)
    ax3.set_title("Optimal Policy for Parameters", fontsize=16)

    # Show the plot
    plt.show()
