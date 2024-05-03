import numpy as np
from src.utils.param_sweep import run_param_sweep
from src.utils.transition_matrix import id_func
from src.utils.small_big import make_smallbig_experiment


def get_start_state(height, width):
    return 0


# Setting the parameters
default_params = dict(
    prob=0.8,
    gamma=0.9,
    height=7,
    width=7,
    big_reward=300,
    small_reward=100,
    latent_reward=0,
)


def perform_sweep(filename=None):
    """
    This world includes a small and a big reward where the small reward is closer to the start state.
    The agent starts in the top-left corner and can move right or down. The small reward is in the bottom-left corner
    and the big reward is in the bottom-right corner. For a 3 by 3 grid, the world looks like this:

    +---+---+---+
    | A |   |   |
    +---+---+---+
    |   |   |   |
    +---+---+---+
    | S |   | B |
    +---+---+---+

    Where A is agent, S is small reward, and B is big reward, with S < B.
    """

    # === Set up the experiment === #
    setup_name = "Big-Small"

    run_parallel = True

    # Set the number of subplots per row
    cols = 4  # 5, 7, 9

    # Set the number of scales and gammas to use
    granularity = 20  # 5, 10, 20

    # Set up parameters to search over
    probs = np.linspace(0.4, 0.99, granularity)
    gammas = np.linspace(0.4, 0.99, granularity)

    search_parameters = {
        # `cols` number of consecutive integers centered around the default value
        "height": np.linspace(3, 11, cols).round().astype(int),
        "width": np.linspace(3, 11, cols).round().astype(int),
        "big_reward": np.linspace(300, 600, cols),
        "small_reward": np.linspace(50, 200, cols),
    }

    rows = len(search_parameters)
    # === End of setup === #

    run_param_sweep(
        setup_name=setup_name,
        default_params=default_params,
        search_parameters=search_parameters,
        create_experiment_func=make_smallbig_experiment,
        transition_matrix_func=id_func,
        rows=rows,
        cols=cols,
        get_start_state=get_start_state,
        granularity=granularity,
        gammas=gammas,
        probs=probs,
        run_parallel=run_parallel,
        subtitle_location=0.95,
        # p2idx_override={
        #     "Close small R": 1,
        #     "Far large R": 0,
        # },
        p2idx_override=None,
        # idx_map={0: 1, 1: 0},
        idx_map=None,
        filename=filename,
    )


if __name__ == "__main__":
    perform_sweep()
