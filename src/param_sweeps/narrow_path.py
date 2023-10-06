from functools import partial
import numpy as np

from src.utils.narrow_path import make_narrow_path_experiment, START
from src.utils.param_sweep import run_param_sweep
from src.utils.enums import Action

make_narrow_path_experiment = partial(
    make_narrow_path_experiment,
    prob=0.5,
    gamma=0.9,
)


def get_start_state(height, width):
    return START


def make_transition_matrix(T, *args, **kwargs):
    return T


def perform_sweep(filename=None):
    # === Start of setup === #
    setup_name = "Narrow Path World"

    run_parallel = True

    # Set the number of subplots per row
    cols = 4  # 5, 7, 9

    # Define the default parameters
    default_params = {
        "big_reward": 100,
        "small_reward": 90,
        "poison_reward": -100,
    }

    # Define the search space
    search_parameters = {
        "small_reward": np.linspace(50, 100, cols),
        "poison_reward": np.linspace(-200, -50, cols),
    }

    # Set the number of scales and gammas to use
    granularity = 20  # 5, 10, 20

    # Set up parameters to search over
    probs = np.linspace(0.01, 0.99, granularity)
    gammas = np.linspace(0.01, 0.99, granularity)

    rows = len(search_parameters)
    # === End of setup === #

    run_param_sweep(
        setup_name=setup_name,
        default_params=default_params,
        search_parameters=search_parameters,
        create_experiment_func=make_narrow_path_experiment,
        transition_matrix_func=make_transition_matrix,
        rows=rows,
        cols=cols,
        get_start_state=get_start_state,
        granularity=granularity,
        probs=probs,
        gammas=gammas,
        run_parallel=run_parallel,
        subtitle_location=0.94,
        show_plot=True,
    )


if __name__ == "__main__":
    perform_sweep()
