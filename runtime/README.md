# Runtime experiments

Code of the two runtime experiments of the paper (Section "Runtime and Edge Performance"), as it ran on an NVIDIA Jetson AGX Thor:

1. **Prediction-stage scaling.** Several prediction workers run FlowChain in parallel on synthetic perception data.
2. **End-to-end multi-intersection inference.** Each intersection has its own pipeline, with a perception process and its own prediction workers, and several pipelines run at the same time.

This folder is self-contained: `src/` holds the scripts and a copy of the FlowChain model code they use. Run every command from this folder (`runtime/`).

| File | Purpose |
|---|---|
| `src/generate_perception.py` | Writes synthetic perception data: a fixed number of objects that move through the image |
| `src/prediction_benchmark.py` | Experiment 1: prediction throughput with parallel workers |
| `src/multi_intersection_pipeline.py` | Experiment 2: several intersection pipelines at the same time |
| `src/camera_processor.py`, `src/prediction_processor.py`, `src/common.py` | The processes of one pipeline: perception, and the pool of prediction workers |
| `config/trajectory.yaml` | Configuration of the FlowChain model |

## Setup

Python packages: `torch`, `numpy`, `pandas`, `scipy`, `dill`, `yacs`, `ncls`, `orjson` and `POT` as for training (see `../requirements.txt`), and in addition `optuna`, `ultralytics`, `opencv-python`, `absl-py` and `lap`. On the Jetson AGX Thor the code ran with Python 3.12, PyTorch 2.14, Ultralytics 8.4.215, TensorRT 10.15 and OpenCV 5.0.

**The forecasting model.** The scripts need a trained FlowChain model and the processed training file of its camera, from which the prediction workers read the standardization constants. A model trained with this repository works directly, for example the Georgia_2 model of a six-intersection run:

```bash
mkdir -p assets src/data/TP/processed_data results
cp ../runs/<run id>/Georgia_2/final_model.pth assets/flowchain.pth
cp ../data/processed_06/Georgia_2_train.pkl src/data/TP/processed_data/
```

`DATASET_NAME` in `config/trajectory.yaml` names this camera (`Georgia_2`); change it when you use the model of another camera. Take the processed file from the folder the model was trained with (`data/processed_06` for a six-intersection run, `data/processed_23` for a 23-view run).

**The perception module.** We use the YOLO26-L detector, converted to a TensorRT engine in FP32 precision, and run it with the ByteTrack tracker from Ultralytics:

```bash
yolo export model=yolo26l.pt format=engine imgsz=640 half=False device=0    # writes yolo26l.engine
mv yolo26l.engine assets/yolo26l_fp32.engine
```

An engine is built for one device, so build it on the device that runs the experiment. The scripts also accept the `.pt` file in place of the engine.

## Experiment 1: prediction-stage scaling

Generate the synthetic perception data, then measure the prediction throughput for several numbers of workers:

```bash
python src/generate_perception.py --n-objects 10 --n-frames 900 \
    --out-pkl data/perception_n10.pkl --out-csv data/perception_n10.csv

python src/prediction_benchmark.py --packets data/perception_n10.pkl \
    --config config/trajectory.yaml --checkpoint assets/flowchain.pth \
    --sweep-workers 1 2 4 6 8 10 12 14 --n-runs 2 --out results/bench_n10.csv
```

- `--n-objects` is the number of objects in every frame (10, 15, 20, 25, 30, 35 and 40 in the paper), and `--n-frames` is the number of frames.
- `--sweep-workers` lists the numbers of parallel workers, and `--n-runs` repeats each of them.
- The output file has one row per number of workers and run. `agg_fps` is the throughput: the number of forecast frames of all workers divided by the run time, without the loading and the warm-up of the models. The other columns give the time of one forecast in milliseconds (`prep` for preparing the input, `infer` for the model call, `total`).
- `--infer-mode per-agent` calls the model once per agent instead of once per frame for all agents, for comparison.
- The folder of `--out` must exist (the `mkdir` command above creates `results`).

## Experiment 2: end-to-end multi-intersection inference

Give one video per intersection. You need your own intersection videos for this experiment; the same file can be given more than once.

```bash
python src/multi_intersection_pipeline.py \
    --videos <video 1> <video 2> --names INT-A INT-B \
    --yolo-model assets/yolo26l_fp32.engine --tracker bytetrack.yaml \
    --config config/trajectory.yaml --checkpoint assets/flowchain.pth \
    --workers 2 --target-fps 30 --save-csv-only --out results/s2_w2
```

- The number of videos is the number of concurrent intersections, and `--workers` is the number of prediction workers of each intersection (one to six intersections with one or two workers in the paper).
- `--tracker bytetrack.yaml` selects ByteTrack; without it Ultralytics uses its default tracker.
- `--target-fps` is the rate at which frames are taken from the video; 30 takes every frame of a 30 FPS video. With `--target-fps 10 --pace arrival --sync-start`, all pipelines start together and each frame is released at the time at which a live 10 FPS camera would deliver it.
- Use a new `--out` folder for every run.

Each run writes `performance_report.csv` into the `--out` folder, with one row per intersection: `camera_avg_fps` is the perception rate, `prediction_avg_fps` the prediction rate and `saved_frames` the number of delivered forecasts. The folder of each intersection holds the forecasts (`all_predictions.csv`) and the time stamps of every forecast (`frames.csv`: arrival of the frame, end of detection, start and end of prediction, delivery). The end-to-end rate is the number of delivered forecasts divided by the time from the start of the stream (`t_stream0` in `camera.json`) to the end of the last forecast (`t_pred_done` in `frames.csv`). On the Jetson, `tegrastats --interval 1000 --logfile <file>` logs memory and power during a run.
