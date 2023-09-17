from src.utils.cafe import make_cafe_experiment
import timeit



if __name__ == "__main__":
    params = {
        "prob": 0.9,
        "gamma": 0.8,
        "vegetarian_reward": 100,
        "donut_reward": 50,
        "noodle_reward": 0,
    }

    def run_solve():
        experiment.mdp.solve(
            show_heatmap=False,
            save_heatmap=False,
        )

    experiment = make_cafe_experiment(**params)

    nruns = 10_000
    total_time = timeit.timeit(run_solve, number=nruns)

    print(f"Average time: {total_time / nruns}, total time for {nruns} runs: {total_time}")

    # Results:
    # - Baseline:
    #   - Average time: 0.0017587373083995772, total time for 10000 runs: 17.58737308399577
    # - Basic Numba
    #   - Average time: 0.0002901063291996252, total time for 10000 runs: 2.901063291996252
    # - Basic Numba + Fastmath
    #   - 
