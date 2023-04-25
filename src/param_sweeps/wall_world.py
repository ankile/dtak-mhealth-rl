import numpy as np
from src.utils.param_sweep import run_param_sweep
from src.utils.transition_matrix import id_func
from src.utils.wall import make_wall_experiment


def get_start_state(height, width):
    return 0


if __name__ == "__main__":
    # === Set up the experiment === #
    setup_name = "Pessimist Wall World"

    run_parallel = True

    # Setting the parameters
    default_params = dict(
        height=5,
        width=6,
        neg_mag=-10,
        latent_cost=0,
        reward_mag=300,
    )

    # Set the number of subplots per row
    cols = 5  # 5, 7, 9

    # Set the number of scales and gammas to use
    granularity = 10  # 5, 10, 20

    # Set up parameters to search over
    scalers, probs = None, None
    # scalers = 2 ** np.linspace(-1, 5, granularity)
    probs = np.linspace(0.4, 0.99, granularity)
    gammas = np.linspace(0.4, 0.99, granularity)

    search_parameters = {
        "reward_mag": np.linspace(100, 500, cols),
        "neg_mag": np.linspace(-20, 0, cols),
        "latent_cost": list(range(-int(cols / 2), int(cols / 2) + 1)),
        # "prob": np.linspace(0.5, 0.95, cols),
        "width": list(range(6 - int(cols / 2), 6 + int(cols / 2) + 1)),
        "height": list(range(6 - int(cols / 2), 6 + int(cols / 2) + 1)),
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
        probs=probs,
        gammas=gammas,
        run_parallel=run_parallel,
    )
