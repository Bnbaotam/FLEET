"""Flower client of the federated strategies: trains and evaluates one camera's model and exchanges parameters with the server."""

import argparse
import os
import time
from typing import Dict, List, Optional

import flwr as fl
import numpy as np
import torch

from data.unified_loader import unified_loader
from fl_train_utils import evaluate_model, fl_local_train
from models.build_model import Build_Model
from strategies.keys import STRATEGIES, exchanged_keys
from utils import load_config, seed_everything

RESULT_KEYS = ("ade", "fde", "ade_legacy", "fde_legacy", "inference_time")  # raw model units / seconds


def train_only(model: torch.nn.Module, keys: List[str], trainable: bool) -> None:
    """requires_grad = True for the parameters in `keys` if trainable, else for the others (FedRep's alternating phases; also the head-only
    adaptation of strategies/local.py); then zero_grad (grads None), so the model's own Adam skips the frozen parameters."""
    for name, param in model.named_parameters():
        param.requires_grad = (name in keys) == trainable
    model.optimizer.zero_grad()


class TrajectoryClient(fl.client.NumPyClient):
    """FedAvg / FedAdam / FedCross client: full-model exchange, one local epoch per round."""

    strategy = "fedavg"

    def __init__(self, cfg, max_batches: Optional[int] = None, save_round_models: bool = False):
        self.cfg = cfg
        self.max_batches = max_batches
        self.save_round_models = save_round_models
        self.model = Build_Model(cfg)
        self.keys = exchanged_keys(self.strategy, self.model)
        self.train_loader = unified_loader(cfg, rand=True, split="train")
        self.test_loader = unified_loader(cfg, rand=False, split="test")
        self.val_loader = unified_loader(cfg, rand=False, split="val") if cfg.DATA.VAL_FRAC > 0 else None

    # ---- parameter exchange -------------------------------------------------
    def get_parameters(self, config) -> List[np.ndarray]:
        state_dict = self.model.state_dict()
        return [state_dict[k].cpu().numpy() for k in self.keys]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        assert len(parameters) == len(self.keys), (len(parameters), len(self.keys))
        state_dict = self.model.state_dict()
        for k, p in zip(self.keys, parameters):
            assert state_dict[k].shape == p.shape, (k, state_dict[k].shape, p.shape)
            state_dict[k] = torch.from_numpy(np.asarray(p))
        self.model.load_state_dict(state_dict, strict=True)

    # ---- local work ---------------------------------------------------------
    def local_train(self, config) -> float:
        return fl_local_train(self.cfg, self.model, self.train_loader, num_epochs=1, max_batches=self.max_batches)

    def local_eval(self, loader, prefix: str = "", return_per_sample: bool = False):
        """Metrics dict for the server (keys prefixed by `prefix`), plus the window table if requested."""
        start = time.perf_counter()
        out = evaluate_model(self.cfg, self.model, loader, max_batches=self.max_batches, return_per_sample=return_per_sample)
        result, table = out if return_per_sample else (out, None)
        metrics = {prefix + k: result[k] for k in RESULT_KEYS}
        metrics[prefix + "n_test_windows"] = result["n_windows"]
        metrics[prefix + "eval_time_s"] = time.perf_counter() - start
        return metrics, table

    def save_round_model(self, config, name: str) -> None:
        """Diagnostic (--save_round_models): state dict as <output_dir>/round_<r>_<name>.pth; reads the model only."""
        if self.save_round_models:
            torch.save(self.model.state_dict(), os.path.join(self.cfg.OUTPUT_DIR, f"round_{int(config['round'])}_{name}.pth"))

    def is_final(self, config) -> bool:
        return int(config["round"]) == int(config["num_rounds"])

    def save_windows(self, table, phase: str) -> None:
        path = os.path.join(self.cfg.OUTPUT_DIR, "per_sample_final.csv")
        table.insert(0, "split", "test")
        table.insert(0, "phase", phase)
        table.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

    # ---- Flower API ---------------------------------------------------------
    def report(self, metrics: Dict) -> Dict:
        return {"intersection": self.cfg.DATA.DATASET_NAME, "n_train_windows": len(self.train_loader.dataset), **metrics}

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        start = time.perf_counter()
        loss = self.local_train(config)
        train_time = time.perf_counter() - start
        self.save_round_model(config, "fit")
        metrics = {}
        if self.is_final(config):  # the fit-phase test evaluation is only used at the final round (FedAvg + fine-tuning baseline)
            metrics, table = self.local_eval(self.test_loader, return_per_sample=True)
            self.save_windows(table, "fit")
        metrics = self.report({**metrics, "loss": loss, "train_time_s": train_time})
        print({"round": int(config["round"]), "phase": "fit", **metrics}, flush=True)
        return self.get_parameters({}), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.save_round_model(config, "global")
        metrics, table = self.local_eval(self.test_loader, return_per_sample=self.is_final(config))
        if self.val_loader is not None:
            metrics.update(self.local_eval(self.val_loader, prefix="val_")[0])
        if table is not None:  # the model held now is the deployable one (FedRep/FedPer: shared encoder + local head)
            self.save_windows(table, "evaluate")
            torch.save(self.model.state_dict(), os.path.join(self.cfg.OUTPUT_DIR, "final_model.pth"))
        metrics = self.report(metrics)
        print({"round": int(config["round"]), "phase": "evaluate", **metrics}, flush=True)
        return metrics["ade"], len(self.test_loader.dataset), metrics


class FedProxClient(TrajectoryClient):
    """Adds mu * (w - w_global) to every local gradient through parameter hooks."""

    strategy = "fedprox"

    def local_train(self, config) -> float:
        proximal_mu = float(config["proximal_mu"])
        assert proximal_mu > 0.0, proximal_mu
        hooks = []
        for _, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            reference = parameter.detach().clone()
            hooks.append(parameter.register_hook(
                lambda gradient, parameter=parameter, reference=reference:
                    gradient + proximal_mu * (parameter.detach() - reference)))
        try:
            return super().local_train(config)
        finally:
            for hook in hooks:
                hook.remove()


class FedRepClient(TrajectoryClient):
    """Exchanges encoder ("representation") only. Per round: 1 epoch head-only, then 1 epoch rep-only."""

    strategy = "fedrep"

    def _train_only(self, trainable: bool) -> None:
        """trainable=True trains the rep (encoder) keys, False trains the head."""
        train_only(self.model, self.keys, trainable)

    def local_train(self, config) -> float:
        self._train_only(trainable=False)
        super().local_train(config)
        self._train_only(trainable=True)
        loss = super().local_train(config)  # the original reports the rep-phase loss only
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.optimizer.zero_grad()
        return loss


class FedPerClient(TrajectoryClient):
    """FedPer (Arivazhagan et al. 2019): joint local training of all parameters, exchanging encoder.* only.

    Identical to the FedAvg client (same fl_local_train path, 1 local epoch per round); flow.* (personal head)
    is never sent and is untouched by set_parameters. The original implementation alternated head/encoder epochs
    instead.
    """

    strategy = "fedper"


CLIENTS = {"fedavg": TrajectoryClient, "fedadam": TrajectoryClient, "fedadambn": TrajectoryClient, "fedcross": TrajectoryClient,
           "fedprox": FedProxClient, "fedrep": FedRepClient, "fedper": FedPerClient}
assert set(CLIENTS) == set(STRATEGIES)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", required=True, choices=STRATEGIES)
    ap.add_argument("--config_file", required=True)
    ap.add_argument("--intersection", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--server_address", required=True)
    ap.add_argument("--val_frac", type=float, required=True, help="fraction of train tracks carved out as validation set (0 = off)")
    ap.add_argument("--max_batches", type=int, default=None, help="smoke tests only: cap train/eval batches")
    ap.add_argument("--save_round_models", action="store_true", help="diagnostic (not used by scripts/run_fl.py): save the received global and the post-fit model every round")
    args = ap.parse_args()

    seed_everything(args.seed)
    cfg = load_config(args.config_file, args.intersection, args.data_dir, args.output_dir, args.seed, args.val_frac)
    client = CLIENTS[args.strategy](cfg, max_batches=args.max_batches, save_round_models=args.save_round_models)
    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
