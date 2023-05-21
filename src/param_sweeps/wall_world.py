import numpy as np
from src.utils.param_sweep import run_param_sweep
from src.utils.transition_matrix import id_func
from src.utils.wall import make_wall_experiment


def get_start_state(height, width):
    return 0


def get_realized_probs_indices(height, width):
    return [1, width - 2, width - 1]


# Setting the parameters
default_params = dict(
    height=4,
    width=6,
    neg_mag=-10,
    reward_mag=200,
)


if __name__ == "__main__":
    # === Set up the experiment === #
    setup_name = "Wall World"

    run_parallel = True

    # Set the number of subplots per row
    cols = 4  # 5, 7, 9

    # Set the number of scales and gammas to use
    granularity = 20  # 5, 10, 20

    # Set up parameters to search over
    scalers, probs = None, None
    # scalers = 2 ** np.linspace(-1, 5, granularity)
    probs = np.linspace(0.4, 0.99, granularity)
    gammas = np.linspace(0.4, 0.99, granularity)

    search_parameters = {
        "height": list(range(4 - int(cols / 2), 4 + int(cols / 2) + 1)),
        "width": np.arange(5, 5 + cols),
        "reward_mag": np.linspace(100, 200, cols),
        "neg_mag": np.linspace(-20, -10, cols),
        # "latent_cost": list(range(-int(cols / 2), int(cols / 2) + 1)),
        # "prob": np.linspace(0.5, 0.95, cols),
    }

    rows = len(search_parameters)

    # === End of setup === #

    run_param_sweep(
        setup_name=setup_name,
        default_params=default_params,
        search_parameters=search_parameters,
        create_experiment_func=make_wall_experiment,
        transition_matrix_func=id_func,
        rows=rows,
        cols=cols,
        get_start_state=get_start_state,
        granularity=granularity,
        gammas=gammas,
        probs=probs,
        # scalers=scalers,
        # get_realized_probs_indices=get_realized_probs_indices,
        run_parallel=run_parallel,
    )
