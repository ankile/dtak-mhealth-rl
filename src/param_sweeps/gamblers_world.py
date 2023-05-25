from functools import partial

import numpy as np

from src.utils.param_sweep import run_param_sweep
from src.utils.gamblers import make_gamblers_experiment, make_gamblers_transition


def get_start_state(height, width):
    return (height * width) // 2


# Setting the parameters
default_params = dict(
    width=5,
    prob=0.8,
    gamma=0.9,
    big_r=10,
    small_r=0,
)


def perform_sweep(filename=None, prob_to_vary="C"):
    """
    Gamblers World:
    """

    assert prob_to_vary in {"C", "F"}, "Can only vary either [C]ontinue or [F]inish"

    # === Set up the experiment === #
    setup_name = f"Gamblers World ($p_{prob_to_vary}$)"

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
        "width": np.linspace(5, 8, cols).round().astype(int),
        "big_r": np.linspace(5, 100, cols).round().astype(int),
        # "small_r": np.linspace(0, 25, cols).round().astype(int),
    }

    rows = len(search_parameters)
    # === End of setup === #

    run_param_sweep(
        setup_name=setup_name,
        default_params=default_params,
        search_parameters=search_parameters,
        create_experiment_func=partial(
            make_gamblers_experiment, vary_continuation=prob_to_vary == "C"
        ),
        transition_matrix_func=partial(
            make_gamblers_transition, vary_continuation=prob_to_vary == "C"
        ),
        rows=rows,
        cols=cols,
        get_start_state=get_start_state,
        granularity=granularity,
        gammas=gammas,
        probs=probs,
        run_parallel=run_parallel,
        filename=filename,
        subtitle_location=0.92,
        p2idx_override={
            "Continue": 0,
            "Finish": 1,
        },
        idx_map={0: 1, 1: 0},
    )


if __name__ == "__main__":
    perform_sweep(prob_to_vary="F")
