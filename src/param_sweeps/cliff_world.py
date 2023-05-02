import numpy as np

from src.utils.cliff import make_cliff_transition, make_cliff_experiment
from src.utils.param_sweep import run_param_sweep
from src.utils.enums import Action


def get_start_state(height, width):
    return (height - 1) * width


def get_realized_probs_indices(height, width):
    # Probability of going up from the bottom left corner
    return [Action.UP.value, (height - 1) * width, (height - 2) * width]


default_params = {
    "height": 5,
    "width": 9,
    "reward_mag": 1e2,
    "small_r_mag": 0,
    "neg_mag": -1e8,
    "latent_reward": 0,
    # These are off by default because they turn the world into a compound world
    "disengage_reward": 0,
    "allow_disengage": False,
}


if __name__ == "__main__":
    # === Start of setup === #
    setup_name = "Cliff World Param Viz"

    run_parallel = True

    # Set the number of subplots per row
    cols = 7  # 5, 7, 9

    # Set the number of scales and gammas to use
    granularity = 20  # 5, 10, 20

    # Set up parameters to search over
    scalers, probs = None, None
    scalers = 2 ** np.linspace(-1, 5, granularity)
    # probs = np.linspace(0.4, 0.99, granularity)
    gammas = np.linspace(0.4, 0.99, granularity)

    search_parameters = {
        "width": np.arange(7 - int(cols / 2), 7 + round(cols / 2)) + 1,
        "height": np.arange(7 - int(cols / 2), 7 + round(cols / 2)) + 1,
        # The rest of these are not interesting to change for different reasons
        # "reward_mag": np.linspace(100, 500, cols),
        # "small_r_mag": np.linspace(25, 100, cols),
        # "neg_mag": np.linspace(-20, 0, cols),
        # "latent_reward": np.linspace(-3, 0, cols),
        # "disengage_reward": np.linspace(0, 10, cols),
        # "prob": np.linspace(0.5, 0.95, cols),  # Don't search over prob
    }

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
        scalers=scalers,
        get_realized_probs_indices=get_realized_probs_indices,
        gammas=gammas,
        run_parallel=run_parallel,
    )
