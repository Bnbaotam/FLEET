"""Local-only baseline worker: one intersection, trained on its own train split only, no communication.
Started by scripts/run_local.py (one process per intersection).

Initialization = the FL server's model (strategies.baselines.fl_initial_state with the run seed); the RNG is then
re-seeded with the FL client's seed (100 * seed + i) before the client is built, exactly as strategies/client.py does.
Per round: 1 local epoch (fl_local_train, same as a FedAvg round), then the test evaluation; one row per round goes to
<output_dir>/rounds_part.jsonl. The final round also writes final_model.pth and per_sample_final.csv.
--init_model PATH (global model + local fine-tuning, run_local.py --init_from) replaces the FL server's init by that
state dict, e.g. python scripts/run_local.py --clients 06 --seed 0 --init_from <fedavg run id> --rounds 5
Worker options that scripts/run_local.py does not use:
--rounds 0 (zero-shot evaluation): one test evaluation of --init_model, logged as round 0, with per_sample_final.csv; no training, no
final_model.pth.
--head_only fedrep|fedper (head-only adaptation): every local epoch trains the head only, i.e. that strategy's shared (exchanged) keys are frozen
as in FedRepClient's head phase (strategies.client.train_only); after every epoch they are asserted bit-identical to the init.
"""
import argparse
import json
import os
import time

import torch

from strategies.baselines import PART, fl_initial_state
from strategies.client import TrajectoryClient, train_only
from strategies.keys import exchanged_keys
from utils import load_config, seed_everything


def frozen_init(model: torch.nn.Module, strategy: str) -> dict:
    """Copies of the model's tensors of the strategy's shared keys, which must all be parameters (the encoder has no buffers)."""
    shared, params = set(exchanged_keys(strategy, model)), {n for n, _ in model.named_parameters()}
    assert shared <= params and not shared & {n for n, _ in model.named_buffers()}, sorted(shared - params)
    state = model.state_dict()
    return {k: state[k].detach().clone() for k in sorted(shared)}


def train_head_only(client: TrajectoryClient, frozen: dict) -> float:
    """One local epoch with the keys of `frozen` frozen (FedRepClient's head phase), then every parameter trainable again (as
    FedRepClient.local_train restores it); asserts the frozen tensors are still bit-identical to `frozen`."""
    train_only(client.model, list(frozen), trainable=False)
    loss = client.local_train({})
    for param in client.model.parameters():
        param.requires_grad = True
    client.model.optimizer.zero_grad()
    state = client.model.state_dict()
    assert all(torch.equal(state[k], v) for k, v in frozen.items()), "a frozen shared tensor changed during head-only training"
    return loss


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config_file", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--intersection", required=True)
    ap.add_argument("--first_intersection", required=True, help="clients[0]: the site the FL server builds its model on")
    ap.add_argument("--init_seed", type=int, required=True, help="the run seed (= the FL server's seed)")
    ap.add_argument("--seed", type=int, required=True, help="this site's seed, 100 * run seed + index")
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--init_model", default=None, help="state dict to start from instead of the FL server's init (fine-tuning)")
    ap.add_argument("--head_only", default=None, choices=["fedrep", "fedper"], help="freeze this strategy's shared keys (train the head only)")
    ap.add_argument("--max_batches", type=int, default=None, help="smoke tests only: cap train/eval batches")
    args = ap.parse_args()
    assert args.rounds > 0 or args.init_model is not None, "zero-shot (--rounds 0) evaluates a given --init_model"
    assert args.head_only is None or (args.rounds > 0 and args.init_model is not None), "--head_only adapts a given --init_model"

    init = fl_initial_state(args.config_file, args.data_dir, args.first_intersection, args.init_seed, args.output_dir) \
        if args.init_model is None else torch.load(args.init_model, map_location="cpu")
    seed_everything(args.seed)
    client = TrajectoryClient(load_config(args.config_file, args.intersection, args.data_dir, args.output_dir, args.seed),
                              max_batches=args.max_batches)
    client.model.load_state_dict(init)
    del init
    frozen = None if args.head_only is None else frozen_init(client.model, args.head_only)
    part = os.path.join(args.output_dir, PART)
    assert not os.path.exists(part), part
    for r in range(1, args.rounds + 1) if args.rounds else [0]:
        start = time.perf_counter()
        loss = None if not r else client.local_train({}) if frozen is None else train_head_only(client, frozen)
        final = r == args.rounds
        metrics, table = client.local_eval(client.test_loader, return_per_sample=final)
        if final:
            client.save_windows(table, "evaluate")
            if r:
                torch.save(client.model.state_dict(), os.path.join(args.output_dir, "final_model.pth"))
        row = {"round": r, "wall_s": time.perf_counter() - start, "metrics": client.report(metrics)}
        print({"train_loss": loss, **row}, flush=True)
        with open(part, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
