import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from src.utils.enums import TransitionMode

from src.utils.transition_matrix import make_absorbing
from src.visualization.worldviz import plot_world_reward
from src.worlds.mdp2d import Experiment_2D


def chain_reward(
    length: int,
    goal_mag: float,
    disengage_reward: float,
    burden: float,
) -> dict:
    """
    Creates a chain of states with a goal state at the end.
    The start state is the first state.
    The goal state is the last state.
    The agent needs to walk through the chain to reach the goal.

    At every state, the agent receives a burden cost.
    Below every state in the chain is a disengage state with a disengage reward.

    :param length: length of the chain
    :param goal_mag: reward for reaching the goal
    :param disengage_mag: reward for disengaging
    :param burden: cost of staying in a state

    returns a dictionary of rewards for each state in the chain.
    """

    # Create the reward dictionary
    reward_dict = {}
    for i in range(length):
        reward_dict[i] = burden  # add burden cost

    # Set the goal state
    reward_dict[length - 1] = goal_mag

    # Set the disengage states
    for i in range(0, length):
        reward_dict[length + i] = disengage_reward

    return reward_dict


def make_chain_transition(T, height, width, prob, params, **kwargs) -> np.ndarray:
    """
    Creates a chain of states with a goal state at the end.
    The start state is the first state.
    The goal state is the last state.
    The agent needs to walk through the chain to reach the goal.

    At every state, the agent receives a burden cost.
    Below every state in the chain is a disengage state with a disengage reward.

    At every state, the agent can choose between action 1 (forward) and action 0 (disengage)
    If the agent chooses action 0, it moves forward with probability prob and stays in place with probability 1 - prob.
    If the agent chooses action 1, it moves to the disengage state with probability params["disengage_prob"],
    moves back to the previous state with probability 1 - params["lost_progress_prob"], and stays in place with
    probability 1 - params["disengage_prob"] - params["lost_progress_prob"].
    Actions 2 and 3 takes the agent back to the same state with probability 1.

    :param T: transition matrix with entries [action, state, next_state]
    :param height: height of the gridworld
    :param width: width of the gridworld
    :param prob: probability of moving forward
    :param kwargs: other arguments

    returns a transition matrix for the chain.
    """

    T = np.zeros_like(T)

    # Set the transition matrix
    for i in range(0, width - 1):
        # Action 1 (forward)
        T[1, i, i + 1] = prob
        T[1, i, i] = 1 - prob

        # Action 3 (disengage)
        T[3, i, i + width] = params["disengage_prob"]
        T[3, i, i] = 1 - params["disengage_prob"] - params["lost_progress_prob"]
        T[3, i, i - 1] = params["lost_progress_prob"]

        # Set the rest of the actions to do nothing
        T[0, i, i] = 1
        T[2, i, i] = 1

    # Set the transition matrix for the last state
    T[0, width - 1, width - 1] = 1
    T[1, width - 1, width - 1] = 1
    T[2, width - 1, width - 1] = 1
    T[3, width - 1, width - 1] = 1

    # Set the transition matrix for the disengage states
    for i in range(0, width):
        T[0, width + i, width + i] = 1
        T[1, width + i, width + i] = 1
        T[2, width + i, width + i] = 1
        T[3, width + i, width + i] = 1

    return T


def make_chain_experiment(
    width=6,
    prob=0.8,
    gamma=0.9,
    goal_mag=10,
    burden=-2,  # For now, burden is the same as lost_progress_cost
    # lost_progress_cost=-1,
    disengage_reward=-0.5,
    disengage_prob=0.1,
    lost_progress_prob=0.3,
) -> Experiment_2D:
    rewards_dict = chain_reward(
        length=width,
        goal_mag=goal_mag,
        disengage_reward=disengage_reward,
        burden=burden,
    )

    experiment = Experiment_2D(
        height=2,
        gamma=gamma,
        width=width,
        rewards_dict=rewards_dict,
        # transition_mode=TransitionMode.FULL,  # We don't need this because the transition matrix is overridden
    )

    T_new = make_chain_transition(
        T=experiment.mdp.T,
        height=experiment.mdp.height,
        width=experiment.mdp.width,
        prob=prob,
        params={
            "disengage_prob": disengage_prob,
            "lost_progress_prob": lost_progress_prob,
        },
    )

    experiment.mdp.T = T_new

    return experiment


if __name__ == "__main__":
    params = {
        "width": 8,
        "prob": 0.72,
        "gamma": 0.8,
        "disengage_prob": 0.9,
        "lost_progress_prob": 0.1,
        "goal_mag": 7,
        "disengage_reward": -3,
        "burden": -2,
    }

    experiment = make_chain_experiment(**params)

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

    # Have no mask for now
    mask = None

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
