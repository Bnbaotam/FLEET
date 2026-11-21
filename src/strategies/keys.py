"""Which FlowChain state-dict entries each strategy exchanges with the server, and the hyperparameter swept per strategy."""
import argparse
from typing import List, Optional
import torch

STRATEGIES = ("fedavg", "fedadam", "fedadambn", "fedprox", "fedcross", "fedper", "fedrep")
# strategy -> (name of the swept hyperparameter [CLI flag --<strategy>_<name>], default value, validation sweep grid).
# The single source of these values: run_fl.py passes the resolved value to the server (FedProx clients get mu from the
# server's round config). The values selected on the validation split are mu = 0.001 (FedProx), eta = 0.1 (FedAdamBN)
# and alpha = 0.5 (FedCross); they are passed explicitly on the command line.
SWEEP = {"fedadam": ("eta", 0.1, (0.1, 0.01, 0.001)),
         "fedadambn": ("eta", 0.1, (0.1, 0.01, 0.001, 0.3)),
         "fedprox": ("mu", 0.01, (0.01, 0.001, 0.1)),
         "fedcross": ("alpha", 0.99, (0.99, 0.9, 0.5))}
assert all(default in grid and len(set(grid)) == len(grid) for _, default, grid in SWEEP.values())


def add_sweep_args(ap: argparse.ArgumentParser) -> None:
    for strategy, (name, default, _) in SWEEP.items():
        ap.add_argument(f"--{strategy}_{name}", type=float, default=None, help=f"{strategy} only (default {default})")


def sweep_value(args: argparse.Namespace) -> Optional[float]:
    """Hyperparameter value of args.strategy (its default if the flag is absent); None for strategies without one."""
    for strategy, (name, _, _) in SWEEP.items():
        assert strategy == args.strategy or getattr(args, f"{strategy}_{name}") is None, f"--{strategy}_{name} needs --strategy {strategy}"
    if args.strategy not in SWEEP:
        return None
    name, default, _ = SWEEP[args.strategy]
    value = getattr(args, f"{args.strategy}_{name}")
    value = default if value is None else value
    assert 0 < value <= (1 if args.strategy == "fedcross" else float("inf")), f"{args.strategy} {name} = {value}"
    return value


def exchanged_keys(strategy: str, model: torch.nn.Module) -> List[str]:
    keys = list(model.state_dict().keys())
    if strategy in ("fedavg", "fedadam", "fedadambn", "fedprox", "fedcross"):
        return keys
    if strategy == "fedrep":
        # selector of the original FedRep client: sorted keys containing "encoder"
        shared = sorted(k for k in keys if "encoder" in k)
    elif strategy == "fedper":
        # selector of the original FedPer client: encoder.* shared, flow.* personal
        shared = [k for k in keys if k.startswith("encoder.")]
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    assert shared and len(shared) < len(keys), (strategy, len(shared), len(keys))
    return shared
