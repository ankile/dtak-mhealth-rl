import os
from matplotlib import pyplot as plt
import numpy as np
from utils.transition_matrix import transition_matrix_is_valid
from worlds.mdp2d import Experiment_2D
from matplotlib.gridspec import GridSpec

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
    small_r_mag,
    neg_mag=-np.inf,
    latent_reward=0,
) -> Experiment_2D:
    if not os.path.exists(savepath := f"local_images/{setup_name}"):
        os.makedirs(savepath)

    cliff_dict = cliff_reward(
        x=width,
        y=height,
        s=neg_mag,
        d=reward_mag,
        latent_reward=latent_reward,
    )

    # hacky solution here
    cliff_dict[width - 1] = small_r_mag

    experiment = Experiment_2D(
        height,
        width,
        rewards_dict=cliff_dict,
        gamma=gamma,
        action_success_prob=prob,
        transition_mode="full",
    )

    assert transition_matrix_is_valid(
        experiment.mdp.T
    ), "whoop whoop Transition matrix is invalid"

    T_new = cliff_transition(
        T=experiment.mdp.T,
        height=height,
        width=width,
    )

    experiment.mdp.T = T_new

    return experiment


if __name__ == "__main__":
    params = {
        "prob": 0.4,
        "gamma": 0.99,
        "height": 4,
        "width": 6,
        "reward_mag": 1000,
        "small_r_mag": 100,
        "neg_mag": -1e4,
        "latent_reward": 0,
    }

    experiment = cliff_experiment(
        setup_name="Cliff_modified",
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

    # # Create a mask for the bottom row if the user is allowed to disengage
    # mask = None
    # if params["allow_disengage"]:
    #     mask = np.zeros(
    #         (params["height"] + int(params["allow_disengage"]), params["width"])
    #     )
    #     mask[-1, :] = 1

    plot_world_reward(experiment, setup_name="Cliff", ax=ax2, show=False)

    experiment.mdp.solve(
        setup_name="Cliff",
        policy_name="Baseline World",
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

    # fig, ax = plt.subplots()

    # # plot_world_reward(experiment, setup_name="Cliff", ax=ax, show=True)

    # experiment.mdp.solve(
    #     setup_name="Cliff_modified",
    #     policy_name="Baseline World",
    #     save_heatmap=True,
    #     base_dir="local_images",
    # )
