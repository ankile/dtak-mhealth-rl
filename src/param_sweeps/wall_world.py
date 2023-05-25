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


def perform_sweep(filename=None):
    # === Set up the experiment === #
    setup_name = "Wall World"

    run_parallel = True

    # Set the number of subplots per row
    cols = 4  # 5, 7, 9

    # Set the number of scales and gammas to use
    granularity = 20  # 5, 10, 20

    # Set up parameters to search over
    probs = np.linspace(0.4, 0.99, granularity)
    gammas = np.linspace(0.4, 0.99, granularity)

    search_parameters = {
        "height": np.linspace(2, 5, cols).round().astype(int),
        "width": np.linspace(5, 8, cols).round().astype(int),
        "reward_mag": np.linspace(100, 250, cols),
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
        run_parallel=run_parallel,
        filename=filename,
        subtitle_location=0.95,
        p2idx_override={
            "Around Wall": 0,
            "Through Wall": 1,
        },
    )


if __name__ == "__main__":
    perform_sweep()
