# FLEET

Federated learning for trajectory forecasting at urban intersections. FLEET trains a FlowChain forecasting model across many traffic cameras with Flower, without pooling their data, and covers the path from tracker output to trained models and error tables.

| Folder | Content |
|---|---|
| `preprocessing/` | Data preparation, from tracker output to the files that training reads |
| `src/` | The forecasting model, data loading, metrics and the federated strategies |
| `scripts/` | Launchers for federated training and the baselines, and the result summaries |
| `configs/` | Model configuration, client lists and standardization constants |
| `data/` | Trajectory data of 10 camera views, ready for training (see `data/README.md`) |
| `runtime/` | Runtime benchmarks of the prediction stage and of the full pipeline (see `runtime/README.md`) |
| `tests/` | Unit tests |

The data files are Python pickles. Load pickles only from sources you trust.

## Quick start

Python 3.10 and one CUDA GPU are required.

```bash
pip install -r requirements.txt        # exact versions: environment/requirements.lock.txt
pytest tests                           # optional: unit tests
python scripts/run_fl.py --strategy fedavg --clients 06 --seed 1 --rounds 2 --max_batches 2   # short test run, a few minutes
```

All commands are run from the repository root. Every script lists its options with `--help`.

## Prepare the data

The repository already holds prepared data, so you can skip this part. It turns the tracker output of a camera into the files that training reads. The input is a text file in MOT format, one box per line: `frame, track_id, x, y, w, h, ...`, with `x, y` the top-left corner of the box in pixels. Trajectories are sampled at 10 frames per second; give the frame rate of the tracker with `--orig-fps`.

```bash
# 1. tracker output of one camera -> raw trajectory files (training and test)
python preprocessing/preprocess_tracks.py --data <tracker output>.txt \
    --train-output data/raw_trajectories/<camera>_train.pkl \
    --test-output  data/raw_trajectories/<camera>_test.pkl

# 2. standardization constants, from the training files of the cameras in a client list
python preprocessing/compute_standardization.py --raw_dir data/raw_trajectories \
    --clients configs/clients_06.txt --out configs/standardization_06.json

# 3. processed files that training reads
python preprocessing/process_data.py --raw_dir data/raw_trajectories --clients configs/clients_06.txt \
    --standardization configs/standardization_06.json --out_dir data/processed_06
```

A client list (`configs/clients_*.txt`) is a text file with one camera name per line.

## Train

One run starts a server and one client process per camera on the same machine.

```bash
python scripts/run_fl.py --strategy fedavg --clients 06 --seed 1
```

| `--strategy` | Method | Its option |
|---|---|---|
| `fedavg` | FedAvg | |
| `fedprox` | FedProx | `--fedprox_mu` |
| `fedadambn` | FedAdam: Adam step on the server, BatchNorm statistics averaged | `--fedadambn_eta` |
| `fedcross` | FedCross | `--fedcross_alpha` |
| `fedper` | FedPer: the encoder is shared, each client keeps its own flow | |
| `fedrep` | FedRep: as FedPer, and each round trains the flow first, then the encoder | |

`--clients 06` and `--clients 23` select the client list `configs/clients_06.txt` or `configs/clients_23.txt`. Other options are `--rounds` (default 20, one local epoch each), `--gpu` and `--port` (two runs on one machine need different ports). The model and optimizer settings are in `configs/flowchain.yml` and `src/default_params.py`.

Baselines:

```bash
python scripts/run_central.py --clients 06 --seed 1                                   # centralized: one model on the pooled training data
python scripts/run_local.py   --clients 06 --seed 1                                   # local-only: one model per camera
python scripts/run_local.py   --clients 06 --seed 1 --init_from <run id> --rounds 5   # a finished federated run, fine-tuned at each camera
```

On one NVIDIA A100, a federated run of 20 rounds took 8 to 12 hours and the centralized baseline 15 to 28 hours.

## Results

Each run writes a folder `runs/<run id>/` with the error of every client after every round, the final model of every client and the logs. To summarize all runs:

```bash
python scripts/summarize.py --clients 06     # ADE and FDE in pixels per camera and method, mean ± standard deviation over the seeds
```

ADE and FDE are computed on the test split of each camera after the last round, from the best of 20 sampled trajectories. Test runs made with `--max_batches` are left out.

## Reproducing the paper

The settings of *FLEET: Federated Learning for Edge-Deployable End-to-End Trajectory Forecasting at Urban Intersections*:

- **Settings.** Six intersections (`--clients 06`) and 23 camera views (`--clients 23`), with seeds 1, 2 and 3. The distributed trajectories were built with `--min-movement 10` in step 1.
- **Hyperparameters.** `--fedprox_mu 0.001`, `--fedadambn_eta 0.1` and `--fedcross_alpha 0.5`, selected on a validation split (`--val_frac 0.1`) at the six intersections.
- **Local-only baseline.** One set of constants per camera: build `data/processed_per_camera` with the two commands below, then run `run_local.py` with `--per_camera` (column `local-pc` of the summary).
- **Fine-tuning.** `--init_from` takes the FedAvg run with the same seed (column `fedavgft`).
- **One scale factor per camera (Appendix A.4).** `python preprocessing/scale_per_camera.py --clients configs/clients_23.txt`, then `run_fl.py` with `--strategy fedavg` or `fedrep`, `--clients 23` and `--scaled`.
- **Views outside the federation (Appendix A.5).** `python preprocessing/unseen_views.py`, then `run_local.py --clients 17 --init_from <run id>` with a six-intersection run and its seed. `--rounds 0` evaluates without training; FedRep and FedPer then take `--head <intersection>`, one run per intersection. `--rounds 5` adapts to the view; FedRep and FedPer then take `--head mean --head_only`. `summarize.py --clients 17` gives the table.
- **Other datasets.** The 23-view setting also uses views of CityFlow and TUMTraf, which are not distributed here (see `data/README.md`).

```bash
python preprocessing/compute_standardization.py --raw_dir data/raw_trajectories \
    --clients configs/clients_23_chattanooga.txt --per_camera --out configs/standardization_per_camera.json
python preprocessing/process_data.py --raw_dir data/raw_trajectories --clients configs/clients_23_chattanooga.txt \
    --standardization configs/standardization_per_camera.json --out_dir data/processed_per_camera
```

## Acknowledgements

The forecasting model in `src/models` and the data structures in `src/data` come from FlowChain (Maeda and Ukita, ICCV 2023, https://github.com/meaten/FlowChain-ICCV2023), which builds on Trajectron++ (Salzmann et al., ECCV 2020). The federated training uses Flower (https://flower.ai), and the runtime benchmarks use Ultralytics (https://github.com/ultralytics/ultralytics).

## License

The code written for FLEET is released under the Apache License 2.0 (see `LICENSE`). The parts taken from FlowChain and Trajectron++ remain under the terms of their authors (see `NOTICE`).
