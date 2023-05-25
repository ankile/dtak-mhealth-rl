from src.param_sweeps.chain_world import perform_sweep as chain_sweep
from src.param_sweeps.cliff_world import perform_sweep as cliff_sweep


if __name__ == "__main__":
    path = lambda s: f"images/plots/parameter_pertubation_{s}.pdf"
    # chain_sweep(path("Chain"))
    cliff_sweep(path("Cliff"))
