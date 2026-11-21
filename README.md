# FLEET

Code and data for the paper *FLEET: Federated Learning for Edge-Deployable End-to-End Trajectory Forecasting at Urban Intersections*.

This repository holds what is needed to go from the output of the perception module to the forecasting results of the paper:

| Folder | Content |
|---|---|
| `data/` | Trajectory data of the Chattanooga camera views: the raw trajectory files and the processed files that training reads |
| `preprocessing/` | The three data-preparation steps: track preprocessing, standardization constants, processed files |
| `configs/` | Model configuration, the client lists of the two settings, and the standardization constants |
| `src/` | The FlowChain forecasting model, data loading, metrics, and the federated learning strategies |
| `scripts/` | Launchers for federated training and the baselines, and the scripts that summarize the results |
| `tests/` | Unit tests of the data preparation, the data loader, the metrics and the strategies |

The paper evaluates two settings. Both use the same code and differ only in the client list:

- **Six intersections** (`--clients 06`): six Chattanooga intersections, one client each.
- **23 camera views** (`--clients 23`): 10 Chattanooga views, 11 CityFlow views and 2 TUMTraf views.

## The process in six steps

```
perception output          one text file per camera, written by the detector and tracker (YOLO26 + ByteTrack)
      |  step 1: preprocess the tracks
raw trajectory files       data/raw_trajectories/<camera>_{train,test}.pkl
      |  step 2: compute the standardization constants
constants                  configs/standardization_<NN>.json
      |  step 3: process the data
processed files            data/processed_<NN>/<camera>_{train,test}.pkl
      |  step 4: train (federated strategies and baselines)
runs                       runs/<run id>/
      |  step 5: summarize
tables                     mean ± standard deviation over the seeds, per camera
```

The repository already contains the outputs of steps 1 to 3 for the Chattanooga views, so you can start at step 4. Steps 1 to 3 are documented so that the files can be rebuilt, and so that the same pipeline can be applied to other cameras.

## Setup

Python 3.10 and one CUDA GPU are required (the model is built on the GPU).

```bash
pip install -r requirements.txt        # environment/requirements.lock.txt is the full list of the versions we used
pytest tests                           # optional: unit tests (a few need the GPU and are skipped without one)
```

All commands below are run from the repository root.

## Step 1: preprocess the tracks

Input: the tracker output of one camera as a text file in MOT format, one line per box:
`frame, track_id, x, y, w, h, ...` with `x, y` the top-left corner of the box in pixels. In the paper, the perception module is the YOLO26 detector followed by the ByteTrack tracker, run on the video of each camera at 10 frames per second.

```bash
python preprocessing/preprocess_tracks.py --data <tracker output>.txt \
    --train-output data/raw_trajectories/<camera>_train.pkl \
    --test-output  data/raw_trajectories/<camera>_test.pkl
```

What it does, in this order:

1. Takes the centre of each box as the position of the object.
2. Marks a track as stationary if it moves less than 10 pixels between its first and its last frame.
3. Fills missing frames inside a track by linear interpolation.
4. Cuts stationary tracks to their first 20 frames.
5. Removes tracks shorter than 20 frames.
6. Splits the tracks into 80 % for training and 20 % for testing. The split is by track, with a fixed seed, so no track is in both sets.

Output: two files per camera, each a table with the columns `frame, trackId, x, y, sceneId, metaId`. The files in `data/raw_trajectories` have exactly this form.

## Step 2: compute the standardization constants

The model works on standardized velocities. The constants are the mean and the standard deviation of position, velocity and acceleration (x and y), computed over the **training** files of all cameras of a setting.

```bash
python preprocessing/compute_standardization.py --raw_dir data/raw_trajectories \
    --clients configs/clients_06.txt --out configs/standardization_06.json
```

The rule: positions are divided by 50; velocity and acceleration are central differences of the position at 0.1 s; the standard deviation is the population value; a negative mean is set to 0; every value is rounded to six decimals.

- `configs/standardization_06.json` holds the constants of the six-intersection setting. The command above reproduces it.
- `configs/standardization_23.json` holds the constants of the 23-view setting. They are computed over all 23 views, so reproducing them needs the CityFlow and TUMTraf trajectory files as well (see "Data").
- Add `--per_camera` to compute one set of constants per camera, each from that camera's own training file. The local-only baseline uses these.

## Step 3: process the data

This step turns the raw trajectory files into the files that training reads: it computes velocities and accelerations, stores the standardization constants in every file, and adds 24 rotated copies of each training scene for augmentation.

```bash
# six-intersection setting
python preprocessing/process_data.py --raw_dir data/raw_trajectories --clients configs/clients_06.txt \
    --standardization configs/standardization_06.json --out_dir data/processed_06

# the Chattanooga views of the 23-view setting
python preprocessing/process_data.py --raw_dir data/raw_trajectories --clients configs/clients_23_chattanooga.txt \
    --standardization configs/standardization_23.json --out_dir data/processed_23

# per-camera constants, for the local-only baseline (one folder serves both settings)
python preprocessing/compute_standardization.py --raw_dir data/raw_trajectories \
    --clients configs/clients_23_chattanooga.txt --per_camera --out configs/standardization_per_camera.json
python preprocessing/process_data.py --raw_dir data/raw_trajectories --clients configs/clients_23_chattanooga.txt \
    --standardization configs/standardization_per_camera.json --out_dir data/processed_per_camera
```

The first two commands reproduce the files that are already in `data/processed_06` and `data/processed_23`. The third pair builds `data/processed_per_camera`, which is not distributed because it follows from the raw trajectory files in a few minutes.

## Step 4: train

One run starts one server and one client process per camera on the same machine. Every run uses 20 rounds with one local epoch per round, batch size 128 and Adam with learning rate 5e-4. The paper reports seeds 1, 2 and 3.

Federated strategies (replace `06` by `23` for the 23-view setting):

```bash
python scripts/run_fl.py --strategy fedavg    --clients 06 --seed 1
python scripts/run_fl.py --strategy fedprox   --clients 06 --seed 1 --fedprox_mu 0.001
python scripts/run_fl.py --strategy fedadambn --clients 06 --seed 1 --fedadambn_eta 0.1
python scripts/run_fl.py --strategy fedcross  --clients 06 --seed 1 --fedcross_alpha 0.5
python scripts/run_fl.py --strategy fedper    --clients 06 --seed 1
python scripts/run_fl.py --strategy fedrep    --clients 06 --seed 1
```

- `fedadambn` is the FedAdam of the paper: the server applies the Adam step to the model parameters and averages the BatchNorm statistics.
- The values of `--fedprox_mu`, `--fedadambn_eta` and `--fedcross_alpha` were tuned on a validation split (10 % of each client's training tracks, `--val_frac 0.1`) at the six intersections and are used unchanged for the 23 views.
- FedPer and FedRep exchange the encoder and keep the normalizing flow at each client. FedRep trains the flow and then the encoder in every round, so it passes over the local data twice.

Baselines:

```bash
python scripts/run_central.py --clients 06 --seed 1                 # centralized: one model on the pooled training data
python scripts/run_local.py   --clients 06 --seed 1 --per_camera    # local-only: one model per camera, per-camera constants (needs data/processed_per_camera)
python scripts/run_local.py   --clients 06 --seed 1 --init_from <run id of the FedAvg run with the same seed> --rounds 5   # FedAvg + local fine-tuning
```

Useful options: `--gpu <id>`, `--port <port>` (two federated runs on one machine need different ports), and `--rounds 2 --max_batches 2` for a quick test of the installation (a few minutes; such runs are tagged `smoke-` and left out of the summaries).

On one NVIDIA A100, a federated run of 20 rounds took 8 to 12 hours in either setting, a local-only run about 8 hours, the centralized run 15 hours (six intersections) or 28 hours (23 views), and fine-tuning about 2 hours.

Each run writes a folder `runs/<run id>/` with the error of every client after every round (`metrics.csv`, `rounds.csv`), the final model of every client and its error on every test window (`<camera>/final_model.pth`, `<camera>/per_sample_final.csv`), and the logs.

## Step 5: summarize

```bash
python scripts/summarize.py --clients 06     # ADE and FDE in pixels per camera, mean ± standard deviation over the seeds
python scripts/summarize.py --clients 23
```

The table has one row per camera and one column per method; `local-pc` is the local-only baseline with per-camera constants and `fedavgft` is FedAvg with local fine-tuning.

ADE and FDE are computed for the model after the last round, on the test split of each camera, from the best of 20 sampled trajectories.

## Data

| Folder | Content |
|---|---|
| `data/raw_trajectories/` | Output of step 1 for the 10 Chattanooga views |
| `data/processed_06/` | Processed files of the six intersections (constants of the six-intersection setting) |
| `data/processed_23/` | Processed files of the 10 Chattanooga views with the constants of the 23-view setting |

`data/README.md` lists the files and the number of tracks and windows per view.

**CityFlow and TUMTraf.** The 23-view setting also uses 11 camera views of CityFlow and 2 of the TUMTraf Intersection dataset (sequence R02_S03). Their trajectory files are not distributed here; the datasets are available from their providers. To get the training and test data of these views, run YOLO26 with ByteTrack on their videos and apply the same data processing: step 1 on the tracker output of each view, then step 3 with `configs/standardization_23.json`. Place the resulting files in `data/processed_23/` under the names listed in `configs/clients_23.txt`. The launchers stop with a list of the missing files if a view is absent.

The data files are Python pickles. Load pickles only from sources you trust.

## Acknowledgements

The forecasting model in `src/models` and the data structures in `src/data` come from FlowChain (Maeda and Ukita, ICCV 2023, https://github.com/meaten/FlowChain-ICCV2023), which builds on Trajectron++ (Salzmann et al., ECCV 2020). The federated training uses Flower (https://flower.ai).

## License

The code written for FLEET is released under the Apache License 2.0 (see `LICENSE`). The parts taken from FlowChain and Trajectron++ remain under the terms of their authors (see `NOTICE`).
