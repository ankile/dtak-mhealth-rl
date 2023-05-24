import numpy as np
from src.utils.param_sweep import run_param_sweep
from src.utils.chain import make_chain_experiment, make_chain_transition


def get_start_state(height, width):
    return 0


# Setting the parameters
default_params = {
    "width": 5,
    "disengage_prob": 0.3,
    "lost_progress_prob": 0.1,
    "goal_mag": 15,
    "disengage_reward": -1,
    "burden": -1,
}


if __name__ == "__main__":
    """
    Chain World

    This is a 1D world where the agent can move left or right. The agent starts at the leftmost state and must reach the
    rightmost state to receive a reward. The agent can also disengage from the task at any point, but this comes with a
    cost. The agent can also lose progress at any point, which also comes with a cost.
    """

    # === Set up the experiment === #
    setup_name = "Chain World"

    run_parallel = True

    # Set the number of subplots per row
    cols = 7  # 5, 7, 9

    # Set the number of scales and gammas to use
    granularity = 20  # 5, 10, 20

    # Set up parameters to search over
    probs = np.linspace(0.4, 0.99, granularity)
    gammas = np.linspace(0.4, 0.99, granularity)

    search_parameters = {
        "width": np.arange(5, 5 + cols),
        "disengage_prob": np.linspace(
            0.1, 0.9, cols
        ),  # Need to make sure this plus below is < 1
        "lost_progress_prob": np.linspace(0.1, 0.7, cols),
        "goal_mag": np.linspace(4, 15, cols),
        "disengage_reward": np.linspace(-1, -10, cols),
        "burden": np.linspace(-0.5, -3, cols),
    }

    rows = len(search_parameters)

    # === End of setup === #

    run_param_sweep(
        setup_name=setup_name,
        default_params=default_params,
        search_parameters=search_parameters,
        create_experiment_func=make_chain_experiment,
        transition_matrix_func=make_chain_transition,
        rows=rows,
        cols=cols,
        get_start_state=get_start_state,
        granularity=granularity,
        gammas=gammas,
        probs=probs,
        run_parallel=run_parallel,
    )
