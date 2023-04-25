import numpy as np

from cliff_world import cliff_experiment
from utils.cliff import cliff_transition
from utils.param_sweep import run_param_sweep

if __name__ == "__main__":
    # === Start of setup ===
    setup_name = "Cliff World Param Viz"

    run_parallel = True

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

    # Set the number of subplots per row
    cols = 8  # 5, 7, 9

    # Set the number of scales and gammas to use
    granularity = 20  # 5, 10, 20

    # Set up parameters to search over
    probs = np.linspace(0.3, 0.99, granularity)
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

    # === End of setup ===

    run_param_sweep(
        setup_name=setup_name,
        default_params=default_params,
        search_parameters=search_parameters,
        create_experiment_func=cliff_experiment,
        transition_matrix_func=cliff_transition,
        rows=rows,
        cols=cols,
        granularity=granularity,
        probs=probs,
        gammas=gammas,
        run_parallel=run_parallel,
    )
