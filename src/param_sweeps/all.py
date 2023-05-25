from src.param_sweeps.chain_world import perform_sweep as chain_sweep
from src.param_sweeps.cliff_world import perform_sweep as cliff_sweep
from src.param_sweeps.gamblers_world import perform_sweep as gamblers_sweep
from src.param_sweeps.riverswim_world import perform_sweep as river_sweep
from src.param_sweeps.smallbig_world import perform_sweep as bigsmall_sweep
from src.param_sweeps.wall_world import perform_sweep as wall_sweep


if __name__ == "__main__":
    path = lambda s: f"images/plots/parameter_pertubation_{s}.pdf"
    # chain_sweep(path("Chain"))
    # cliff_sweep(path("Cliff"))
    # gamblers_sweep(path("Gamblers_pC"))
    # gamblers_sweep(path("Gamblers_pF"), prob_to_vary="F")
    # river_sweep(path("RiverSwim"))
    bigsmall_sweep(path("BigSmall"))
    # wall_sweep(path("Wall"))
