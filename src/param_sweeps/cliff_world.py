import numpy as np

from src.utils.cliff import make_cliff_transition, make_cliff_experiment
from src.utils.param_sweep import run_param_sweep
from src.utils.enums import Action


def get_start_state(height, width):
    return (height - 1) * width



def perform_sweep(filename=None):
    # === Start of setup === #
    setup_name = "Cliff World"

    run_parallel = True

    # Set the number of subplots per row
    cols = 4  # 5, 7, 9

    # Define the default parameters
    default_params = {
        "height": 5,
        "width": 9,
        "reward_mag": 1e2,
        "neg_mag": -1e8,
        "latent_reward": 0,
    }

    # Define the search space
    search_parameters = {
        "width": np.linspace(4, 10, cols).round().astype(int),
        "height": np.linspace(4, 10, cols).round().astype(int),
        "reward_mag": np.linspace(100, 500, cols),
    }

    # Set the number of scales and gammas to use
    granularity = 10  # 5, 10, 20

    # Set up parameters to search over
    probs = np.linspace(0.4, 0.99, granularity)
    gammas = np.linspace(0.4, 0.99, granularity)


    rows = len(search_parameters)
    # === End of setup === #

    run_param_sweep(
        setup_name=setup_name,
        default_params=default_params,
        search_parameters=search_parameters,
        create_experiment_func=make_cliff_experiment,
        transition_matrix_func=make_cliff_transition,
        rows=rows,
        cols=cols,
        get_start_state=get_start_state,
        granularity=granularity,
        probs=probs,
        gammas=gammas,
        run_parallel=run_parallel,
        subtitle_location=0.94,
        p2idx_override={
            "Safe": 0,
            "Dangerous": 1,
        },
        idx_map={
            0: 1,
        }
        | {i: 0 for i in range(1, 10)},
        filename=filename,
    )


if __name__ == "__main__":
    perform_sweep()
