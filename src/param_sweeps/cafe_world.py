import numpy as np
from src.utils.param_sweep import run_param_sweep
from src.utils.cafe import c2s, make_cafe_experiment, make_cafe_transition


def get_start_state(height, width):
    return c2s(12, 3)


# Setting the parameters
default_params = dict(
    vegetarian_reward=200,
    donut_reward=50,
    noodle_reward=100,
)


def perform_sweep(filename=None):
    # === Set up the experiment === #
    setup_name = "Cafe World"

    run_parallel = True

    # Set the number of subplots per row
    cols = 4  # 5, 7, 9, or 4 for the publication

    # Set the number of scales and gammas to use
    granularity = 20  # 5, 10, 20

    # Set up parameters to search over
    probs = np.linspace(0.4, 0.99, granularity)
    gammas = np.linspace(0.4, 0.99, granularity)

    search_parameters = {
        "vegetarian_reward": np.linspace(100, 250, cols),
        "donut_reward": np.linspace(25, 75, cols),
        "noodle_reward": np.linspace(50, 150, cols),
    }

    rows = len(search_parameters)
    # === End of setup === #

    run_param_sweep(
        setup_name=setup_name,
        default_params=default_params,
        search_parameters=search_parameters,
        create_experiment_func=make_cafe_experiment,
        transition_matrix_func=make_cafe_transition,
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
            "Donut": 0,
            "Vegan/Noodle": 1,
        },
        idx_map={
            0: 0,
            1: 1,
            2: 1,
        },
    )


if __name__ == "__main__":
    perform_sweep("images/plots/parameter_pertubation_Cafe.pdf")
