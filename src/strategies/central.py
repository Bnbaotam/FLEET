"""Centralized baseline worker: one model trained on the pooled train splits of every intersection (not federated).
Started by scripts/run_central.py.

Initialization = the FL server's model (fl_initial_state with the run seed); training RNG seed = 100 * seed (that of client 0).
Pooled train set = ConcatDataset of every site's train dataset, shuffled, batch size / drop_last / collate as the train loaders.
Per epoch: fl_local_train for 1 epoch (same path and fresh outer Adam as a FedAvg round). Every `eval_every` epochs and at the
last one, the single model is evaluated on every site's test split (the cost of one evaluation is that of all sites, hence not
every epoch) and one "evaluate" row per site goes through RoundLogger. The last epoch writes, for every site, per_sample_final.csv
and a copy of the one final_model.pth in <run_dir>/<site>/.
"""
import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import ConcatDataset

from data.unified_loader import make_loader
from fl_train_utils import fl_local_train
from strategies.baselines import assert_same_standardization, fl_initial_state
from strategies.client import TrajectoryClient
from strategies.metrics_log import RoundLogger
from utils import load_config, seed_everything


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config_file", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", required=True, help="logs only; per-site outputs go to <run_dir>/<site>/")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--sites", nargs="+", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rounds", type=int, required=True, help="epochs")
    ap.add_argument("--eval_every", type=int, required=True)
    ap.add_argument("--max_batches", type=int, default=None, help="smoke tests only: cap train/eval batches")
    args = ap.parse_args()
    assert args.eval_every >= 1

    run_dir = Path(args.run_dir)
    init = fl_initial_state(args.config_file, args.data_dir, args.sites[0], args.seed, args.output_dir)
    seed_everything(100 * args.seed)
    clients = [TrajectoryClient(load_config(args.config_file, s, args.data_dir, str(run_dir / s), args.seed),
                                max_batches=args.max_batches) for s in args.sites]
    model = clients[0].model
    model.load_state_dict(init)
    del init
    for c in clients:  # every site evaluates (and saves) the one trained model
        c.model = model
    assert_same_standardization([c.train_loader.dataset.env for c in clients])
    cfg = clients[0].cfg
    loader = make_loader(cfg, ConcatDataset([c.train_loader.dataset for c in clients]), rand=True, drop_last=True)
    print(f"pooled train windows: {len(loader.dataset)} (sites: {[len(c.train_loader.dataset) for c in clients]})", flush=True)
    logger = RoundLogger(args.run_dir, args.run_id, "central", len(clients), args.seed, args.rounds, 0.0)

    for epoch in range(1, args.rounds + 1):
        start = time.perf_counter()
        loss = fl_local_train(cfg, model, loader, num_epochs=1, max_batches=args.max_batches)
        final = epoch == args.rounds
        if not (final or epoch % args.eval_every == 0):
            print({"epoch": epoch, "train_loss": loss}, flush=True)
            continue
        results = []
        for c in clients:
            metrics, table = c.local_eval(c.test_loader, return_per_sample=final)
            if final:
                c.save_windows(table, "evaluate")
                torch.save(model.state_dict(), Path(c.cfg.OUTPUT_DIR) / "final_model.pth")
            results.append(SimpleNamespace(metrics=c.report(metrics)))
        logger.log("evaluate", epoch, results, time.perf_counter() - start, 0.0)
        print({"epoch": epoch, "train_loss": loss, "ade_px_mean": [r.metrics["ade"] for r in results]}, flush=True)


if __name__ == "__main__":
    main()
