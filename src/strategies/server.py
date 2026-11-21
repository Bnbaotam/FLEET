"""Flower server for all strategies; started by scripts/run_fl.py.

Fixed hyperparameters are those of the original implementation (FEDADAM, FEDCROSS below); the swept ones are in
strategies/keys.py. The original FedRep and FedPer servers were custom strategies that compute a sample-weighted average
of whatever the clients send, i.e. FedAvg aggregation over the shared keys only, so they use FedAvg here.

fedadambn: FedAdam whose server Adam step moves the trainable parameters only, while the buffers take the FedAvg average
(FedAdamBN below); this is the FedAdam variant whose results the paper reports. fedadam: Flower's stock FedAdam, which also
Adam-steps the BatchNorm running statistics (used by the flow in eval mode) and can drive running_var negative.

Every strategy starts from server-built initial parameters (seeded). In the original implementation FedAvg, FedProx and
FedRep took the initial parameters from one randomly chosen client.
"""
import argparse
import time
from datetime import timedelta
from typing import Optional

import flwr as fl
import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from models.build_model import Build_Model
from strategies.fedcross import FedCrossStrategy
from strategies.keys import STRATEGIES, SWEEP, exchanged_keys
from strategies.metrics_log import RoundLogger
from utils import load_config, seed_everything

# eta (FedAdam), mu (FedProx) and alpha (FedCross) are in keys.SWEEP: defaults, overridable per run (run_fl.py --<strategy>_<name>)
FEDADAM = dict(eta_l=0.1, beta_1=0.9, beta_2=0.99, tau=1e-3)
FEDCROSS = dict(first_stage_bound=0, collaborative_model_select_strategy=1)


class FedAdamBN(fl.server.strategy.FedAdam):
    """FedAdam with the arrays at `buffer_idx` (model buffers) set to the sample-weighted client average instead of the Adam step,
    both in the returned parameters and in the server state (current_weights) the next round's update starts from."""

    def __init__(self, *, buffer_idx, running_var_idx, **kwargs):
        super().__init__(**kwargs)
        assert buffer_idx and set(running_var_idx) <= set(buffer_idx), (buffer_idx, running_var_idx)
        self.buffer_idx, self.running_var_idx = buffer_idx, running_var_idx

    def aggregate_fit(self, server_round, results, failures):
        assert results and not failures, failures
        avg = parameters_to_ndarrays(fl.server.strategy.FedAvg.aggregate_fit(self, server_round, results, failures)[0])
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        out = parameters_to_ndarrays(parameters)
        assert len(out) == len(avg) == len(self.current_weights), (len(out), len(avg))
        for i in self.buffer_idx:
            out[i] = self.current_weights[i] = avg[i]
        assert all((avg[i] >= 0).all() for i in self.running_var_idx) and all(np.isfinite(x).all() for x in out)
        return ndarrays_to_parameters(out), metrics


def with_logging(base, logger: RoundLogger):
    """Subclass a Flower strategy (any of them) so that fit/evaluate configs carry the round numbers and every
    round's per-client metrics go through `logger` (asserting no failures and one report per client)."""

    class Strategy(base):
        phase_start = {}

        def configure_fit(self, server_round, parameters, client_manager):
            self.phase_start["fit"] = time.time()
            return [(c, fl.common.FitIns(ins.parameters, {**ins.config, "round": server_round, "num_rounds": logger.num_rounds}))
                    for c, ins in super().configure_fit(server_round, parameters, client_manager)]

        def configure_evaluate(self, server_round, parameters, client_manager):
            self.phase_start["evaluate"] = time.time()
            return [(c, fl.common.EvaluateIns(ins.parameters, {**ins.config, "round": server_round, "num_rounds": logger.num_rounds}))
                    for c, ins in super().configure_evaluate(server_round, parameters, client_manager)]

        def timed(self, phase, aggregate, server_round, results, failures):
            assert not failures, failures
            start = time.time()
            out = aggregate(server_round, results, failures)
            end = time.time()
            logger.log(phase, server_round, [r for _, r in results], end - self.phase_start[phase], end - start)
            return out

        def aggregate_fit(self, server_round, results, failures):
            return self.timed("fit", super().aggregate_fit, server_round, results, failures)

        def aggregate_evaluate(self, server_round, results, failures):
            return self.timed("evaluate", super().aggregate_evaluate, server_round, results, failures)

    Strategy.__name__ = f"Logged{base.__name__}"
    return Strategy


def build_strategy(name: str, model, num_clients: int, num_rounds: int, logger: RoundLogger, sweep_value: Optional[float]):
    assert (sweep_value is None) == (name not in SWEEP), (name, sweep_value)
    keys = exchanged_keys(name, model)
    state_dict = model.state_dict()
    initial = fl.common.ndarrays_to_parameters([state_dict[k].detach().cpu().numpy() for k in keys])
    clients = dict(min_fit_clients=num_clients, min_evaluate_clients=num_clients, min_available_clients=num_clients)
    common = dict(fraction_fit=1.0, fraction_evaluate=1.0, initial_parameters=initial, **clients)

    if name in ("fedavg", "fedrep", "fedper"):
        return with_logging(fl.server.strategy.FedAvg, logger)(**common)
    if name == "fedadam":
        return with_logging(fl.server.strategy.FedAdam, logger)(**common, **FEDADAM, eta=sweep_value)
    if name == "fedadambn":
        buffers, params = {k for k, _ in model.named_buffers()}, {k for k, _ in model.named_parameters()}
        assert set(keys) == params | buffers and not params & buffers, (len(keys), len(params), len(buffers))
        assert {k.rsplit(".", 1)[-1] for k in buffers} == {"mask", "running_mean", "running_var"}, buffers
        return with_logging(FedAdamBN, logger)(**common, **FEDADAM, eta=sweep_value, buffer_idx=[i for i, k in enumerate(keys) if k in buffers],
                                               running_var_idx=[i for i, k in enumerate(keys) if k.endswith(".running_var")])
    if name == "fedprox":
        return with_logging(fl.server.strategy.FedProx, logger)(**common, proximal_mu=sweep_value)
    if name == "fedcross":
        return with_logging(FedCrossStrategy, logger)(initial_parameters=initial, model_keys=keys, num_rounds=num_rounds, **clients, **FEDCROSS, fedcross_alpha=sweep_value)
    raise ValueError(f"unknown strategy: {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", required=True, choices=STRATEGIES)
    ap.add_argument("--config_file", required=True)
    ap.add_argument("--intersection", required=True, help="any client; only used to build the model for its shapes")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--run_dir", required=True, help="where metrics.csv and rounds.csv are written")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--val_frac", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--num_clients", type=int, required=True)
    ap.add_argument("--num_rounds", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--sweep_value", type=float, default=None, help="FedAdam eta / FedProx mu / FedCross alpha (required for exactly these)")
    args = ap.parse_args()

    seed_everything(args.seed)
    cfg = load_config(args.config_file, args.intersection, args.data_dir, args.output_dir, args.seed)
    logger = RoundLogger(args.run_dir, args.run_id, args.strategy, args.num_clients, args.seed, args.num_rounds, args.val_frac)
    strategy = build_strategy(args.strategy, Build_Model(cfg), args.num_clients, args.num_rounds, logger, args.sweep_value)

    start = time.time()
    fl.server.start_server(server_address=f"127.0.0.1:{args.port}",
                           config=fl.server.ServerConfig(num_rounds=args.num_rounds), strategy=strategy)
    print(f"Total runtime: {timedelta(seconds=int(time.time() - start))} (HH:MM:SS)", flush=True)


if __name__ == "__main__":
    main()
