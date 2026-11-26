# Runtime benchmarks

Two benchmarks of FLEET at inference time. This folder is self-contained: `src/` holds the scripts and a copy of the FlowChain model code they use. Run every command from this folder (`runtime/`).

1. **Prediction throughput** (`src/prediction_benchmark.py`): several prediction workers run FlowChain in parallel on synthetic perception data.
2. **Multi-intersection pipeline** (`src/multi_intersection_pipeline.py`): each intersection has its own pipeline of detection, tracking and prediction workers, and several pipelines run at the same time.

## Setup

Install the packages of `../requirements.txt`, and in addition `optuna`, `ultralytics`, `opencv-python`, `absl-py` and `lap`. The benchmarks were run on an NVIDIA Jetson AGX Thor with Python 3.12, PyTorch 2.14, Ultralytics 8.4.215, TensorRT 10.15 and OpenCV 5.0.

The scripts need a trained FlowChain model and the processed training file of its camera, which holds the standardization constants:

```bash
mkdir -p assets src/data/TP/processed_data results
cp ../runs/<run id>/Georgia_2/final_model.pth assets/flowchain.pth
cp ../data/processed_06/Georgia_2_train.pkl src/data/TP/processed_data/
```

`DATASET_NAME` in `config/trajectory.yaml` names this camera. Change it when you use the model of another camera, and take the processed file from the folder the model was trained with.

The second benchmark also needs a detector. We use YOLO26-L as a TensorRT engine, built on the device that runs the benchmark (the scripts also accept the `.pt` file):

```bash
yolo export model=yolo26l.pt format=engine imgsz=640 half=False device=0    # writes yolo26l.engine
mv yolo26l.engine assets/yolo26l_fp32.engine
```

## 1. Prediction throughput

```bash
python src/generate_perception.py --n-objects 10 --n-frames 900 \
    --out-pkl data/perception_n10.pkl --out-csv data/perception_n10.csv

python src/prediction_benchmark.py --packets data/perception_n10.pkl \
    --config config/trajectory.yaml --checkpoint assets/flowchain.pth \
    --sweep-workers 1 2 4 6 8 10 12 14 --n-runs 2 --out results/bench_n10.csv
```

`--n-objects` is the number of objects in every frame, `--sweep-workers` lists the numbers of parallel workers and `--n-runs` repeats each of them. The folder of `--out` must exist. The output has one row per number of workers and run: `agg_fps` is the throughput (the forecast frames of all workers divided by the run time, without the loading and the warm-up of the models), and the other columns give the time of one forecast in milliseconds.

## 2. Multi-intersection pipeline

Give one video per intersection; the same file can be given more than once.

```bash
python src/multi_intersection_pipeline.py \
    --videos <video 1> <video 2> --names INT-A INT-B \
    --yolo-model assets/yolo26l_fp32.engine --tracker bytetrack.yaml \
    --config config/trajectory.yaml --checkpoint assets/flowchain.pth \
    --workers 2 --target-fps 30 --save-csv-only --out results/s2_w2
```

`--workers` is the number of prediction workers of each intersection, and `--target-fps` the rate at which frames are taken from the video. With `--target-fps 10 --pace arrival --sync-start`, all pipelines start together and each frame is released when a live 10 FPS camera would deliver it. Use a new `--out` folder for every run.

Each run writes `performance_report.csv` with one row per intersection (`camera_avg_fps`, `prediction_avg_fps`, `saved_frames`), and one folder per intersection with the forecasts (`all_predictions.csv`) and the time stamps of every forecast (`frames.csv`).

## Settings of the paper

The first benchmark uses 10 to 40 objects in steps of 5, and the second one to six intersections with one or two workers each. The end-to-end rate is the number of delivered forecasts divided by the time from the start of the stream (`t_stream0` in `camera.json`) to the end of the last forecast (`t_pred_done` in `frames.csv`).
