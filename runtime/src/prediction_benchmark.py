#!/usr/bin/env python3
"""Measures the throughput of FlowChain trajectory prediction with parallel workers on stored perception data."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_LOGGING_STDERR_THRESHOLD", "3")

import sys
import time
import pickle
import itertools
import argparse
import traceback
import multiprocessing as mp
from copy import deepcopy
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

# ── same imports as current prediction_processor.py ───────────────────────────
from data.TP.environment import Environment, Node, Scene, derivative_of
from data.TP.trajectron_dataset import EnvironmentDataset, hypers
from data.TP.preprocessing import dict_collate as seq_collate
from utils import load_config
from models.build_model import Build_Model
from metrics.build_metrics import Build_Metrics

T_O = 8
T_F = 12  # kept for compatibility / benchmark format

FALLBACK_STANDARDIZATION = {
    "PEDESTRIAN": {
        "position": {
            "x": {"mean": 17.82, "std": 5.52},
            "y": {"mean": 8.93, "std": 1.44},
        },
        "velocity": {
            "x": {"mean": 0.25, "std": 2.14},
            "y": {"mean": 0.07, "std": 0.72},
        },
        "acceleration": {
            "x": {"mean": 0.04, "std": 6.93},
            "y": {"mean": 0.12, "std": 3.10},
        },
    }
}


# ──────────────────────────────────────────────────────────────────────────────
# Current prediction core helpers (ported from prediction_processor.py)
# Keep benchmark format; no class filtering.
# ──────────────────────────────────────────────────────────────────────────────
def _get_col(df: pd.DataFrame, *names: str) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def _to_int_track_id(v) -> Optional[int]:
    try:
        if pd.isna(v):
            return None
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return None


def _get_standardization(metrics) -> dict:
    train_env = getattr(metrics, "env", None)
    if train_env is not None and getattr(train_env, "standardization", None):
        return deepcopy(train_env.standardization)
    return deepcopy(FALLBACK_STANDARDIZATION)


def _process_track_data(
    arr_xy: np.ndarray,
    scale: float,
    dt: float,
    data_columns: pd.MultiIndex,
) -> Optional[pd.DataFrame]:
    try:
        arr = np.asarray(arr_xy, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            return None

        x = arr[:, 0].astype(float) / scale
        y = arr[:, 1].astype(float) / scale

        vx = derivative_of(x, dt) if len(x) > 1 else np.zeros_like(x)
        vy = derivative_of(y, dt) if len(y) > 1 else np.zeros_like(y)
        ax = derivative_of(vx, dt) if len(vx) > 1 else np.zeros_like(vx)
        ay = derivative_of(vy, dt) if len(vy) > 1 else np.zeros_like(vy)

        data_dict = {
            ("position", "x"): x,
            ("position", "y"): y,
            ("velocity", "x"): vx,
            ("velocity", "y"): vy,
            ("acceleration", "x"): ax,
            ("acceleration", "y"): ay,
        }
        return pd.DataFrame(data_dict, columns=data_columns)
    except Exception:
        return None


def _build_track_histories(
    df_window: pd.DataFrame,
    latest_df: pd.DataFrame,
    latest_fidx: int,
    obs_len: int,
) -> Dict[int, np.ndarray]:
    """
    Build fixed-length x,y histories for tracks active in the latest frame.
    Missing timesteps inside the observation window are linearly interpolated.
    """
    track_col = _get_col(df_window, "trackId")
    frame_col = _get_col(df_window, "frame", "frame_idx")
    x_col = _get_col(df_window, "x")
    y_col = _get_col(df_window, "y")

    if not all([track_col, frame_col, x_col, y_col]):
        return {}

    latest_track_col = _get_col(latest_df, "trackId")
    if latest_track_col is None:
        return {}

    active_ids = []
    for v in latest_df[latest_track_col].tolist():
        tid = _to_int_track_id(v)
        if tid is not None:
            active_ids.append(tid)

    if not active_ids:
        return {}

    frame_range = np.arange(latest_fidx - obs_len + 1, latest_fidx + 1, dtype=int)
    histories: Dict[int, np.ndarray] = {}

    dfw = df_window.copy()
    dfw[track_col] = pd.to_numeric(dfw[track_col], errors="coerce")
    dfw[frame_col] = pd.to_numeric(dfw[frame_col], errors="coerce")
    dfw[x_col] = pd.to_numeric(dfw[x_col], errors="coerce")
    dfw[y_col] = pd.to_numeric(dfw[y_col], errors="coerce")
    dfw = dfw.dropna(subset=[track_col, frame_col, x_col, y_col])

    for tid in sorted(set(active_ids)):
        g = dfw[dfw[track_col] == tid].copy()
        if g.empty:
            continue

        g = g.sort_values(frame_col)
        g = g.drop_duplicates(subset=[frame_col], keep="last")
        g = g[[frame_col, x_col, y_col]].rename(
            columns={frame_col: "frame", x_col: "x", y_col: "y"}
        )
        g["frame"] = g["frame"].astype(int)
        g = g.set_index("frame").reindex(frame_range)

        if latest_fidx not in g.index:
            continue

        if pd.isna(g.loc[latest_fidx, "x"]) or pd.isna(g.loc[latest_fidx, "y"]):
            continue

        g[["x", "y"]] = g[["x", "y"]].interpolate(
            method="linear",
            limit_direction="both",
        )

        if g[["x", "y"]].isna().any().any():
            continue

        histories[tid] = g[["x", "y"]].to_numpy(dtype=float)

    return histories


def _create_env_from_histories(
    track_histories: Dict[int, np.ndarray],
    dt: float,
    scale: float,
    standardization: dict,
) -> Optional[Environment]:
    if not track_histories:
        return None

    env = Environment(
        node_type_list=["PEDESTRIAN"],
        standardization=standardization,
    )
    env.attention_radius = {
        (env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN): 15.0,
    }

    scene = Scene(
        timesteps=T_O,
        dt=dt,
        name="benchmark",
    )

    data_columns = pd.MultiIndex.from_product(
        [["position", "velocity", "acceleration"], ["x", "y"]]
    )

    for track_id, arr_xy in track_histories.items():
        node_data = _process_track_data(
            arr_xy=arr_xy,
            scale=scale,
            dt=dt,
            data_columns=data_columns,
        )
        if node_data is None:
            continue

        node = Node(
            node_type=env.NodeType.PEDESTRIAN,
            node_id=str(track_id),
            data=node_data,
        )
        node.first_timestep = 0
        scene.nodes.append(node)

    if not scene.nodes:
        return None

    env.scenes = [scene]
    return env


# ──────────────────────────────────────────────────────────────────────────────
# Same benchmark interface, new prediction core
# ──────────────────────────────────────────────────────────────────────────────
def _predict_window(
    window_packets: List[Dict[str, Any]],
    cfg,
    model,
    metrics,
    device: torch.device,
    infer_mode: str = "batched",
) -> Dict[str, float]:
    """
    Feed T_O camera packets through the full prediction pipeline.
    Returns timing dict: prep_ms, infer_ms, post_ms, total_ms.
    Returns all zeros if no valid prediction was possible.
    """
    timing = {"prep_ms": 0.0, "infer_ms": 0.0, "post_ms": 0.0, "total_ms": 0.0}
    t_total = time.perf_counter()

    # ── 1) Prep (same current logic as prediction_processor.py) ────────────
    t0 = time.perf_counter()

    latest_pkt = window_packets[-1]
    latest_fidx = int(latest_pkt["frame_idx"])
    latest_df = latest_pkt.get("perception_df")

    if latest_df is None or (isinstance(latest_df, pd.DataFrame) and latest_df.empty):
        return timing

    frames = []
    for p in window_packets:
        dfp = p.get("perception_df")
        if dfp is not None and isinstance(dfp, pd.DataFrame) and not dfp.empty:
            frames.append(dfp)

    if not frames:
        return timing

    df_window = pd.concat(frames, ignore_index=True)
    if df_window.empty:
        return timing

    scale = float(getattr(cfg.DATA, "SCALE", 50.0))
    dt = float(getattr(cfg.DATA, "DT", 0.1))
    standardization = _get_standardization(metrics)

    track_histories = _build_track_histories(
        df_window=df_window,
        latest_df=latest_df,
        latest_fidx=latest_fidx,
        obs_len=T_O,
    )
    if not track_histories:
        return timing

    env = _create_env_from_histories(
        track_histories=track_histories,
        dt=dt,
        scale=scale,
        standardization=standardization,
    )
    if env is None:
        return timing

    dataset = EnvironmentDataset(
        env,
        state=hypers[cfg.DATA.TP.STATE],
        pred_state=hypers[cfg.DATA.TP.PRED_STATE],
        node_freq_mult=hypers["scene_freq_mult_train"],
        scene_freq_mult=hypers["node_freq_mult_train"],
        hyperparams=hypers,
        min_history_timesteps=T_O - 1,
        min_future_timesteps=0,
    )

    pedestrian_dataset = None
    for node_type_dataset in dataset:
        if getattr(node_type_dataset, "node_type", None) == "PEDESTRIAN":
            pedestrian_dataset = node_type_dataset
            break
        if len(node_type_dataset) > 0 and pedestrian_dataset is None:
            pedestrian_dataset = node_type_dataset

    if pedestrian_dataset is None or len(pedestrian_dataset) == 0:
        return timing

    assert infer_mode in ("batched", "per-agent"), infer_mode
    n_batched = min(50, max(1, len(pedestrian_dataset)))   # the batched mode predicts the first batch only
    loader = DataLoader(
        pedestrian_dataset,
        batch_size=n_batched if infer_mode == "batched" else 1,
        shuffle=False,
        num_workers=0,
        collate_fn=seq_collate,
        drop_last=False,
        pin_memory=False,
    )

    # added: per-agent = the same n_batched agents, one model.predict call each
    batches = list(itertools.islice(loader, 1 if infer_mode == "batched" else n_batched))
    if not batches:
        return timing

    data_dicts = [{
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    } for batch in batches]

    timing["prep_ms"] = (time.perf_counter() - t0) * 1e3

    # ── 2) Inference ───────────────────────────────────────────────────────
    t1 = time.perf_counter()
    with torch.no_grad():
        results = [metrics.denormalize([model.predict(deepcopy(d), return_prob=False)])[0]
                   for d in data_dicts]
    timing["infer_ms"] = (time.perf_counter() - t1) * 1e3

    # ── 3) Post-process ────────────────────────────────────────────────────
    t2 = time.perf_counter()
    pred_map: Dict[int, np.ndarray] = {}
    for data_dict, result_denorm in zip(data_dicts, results):
        for i, pred in enumerate(result_denorm[("pred", 0)]):
            raw_tid = data_dict["index"][i][2].split("/")[-1]
            tid = _to_int_track_id(raw_tid)
            if tid is None:
                continue
            pred_map[tid] = pred.detach().cpu().numpy() * scale
    assert len(pred_map) == n_batched, (len(pred_map), n_batched)
    timing["n_agents"] = n_batched

    timing["post_ms"] = (time.perf_counter() - t2) * 1e3
    timing["total_ms"] = (time.perf_counter() - t_total) * 1e3
    return timing


# ──────────────────────────────────────────────────────────────────────────────
# Worker subprocess
# ──────────────────────────────────────────────────────────────────────────────
def _load_model(cfg_path, ckpt_path, device):
    cfg = load_config(argparse.Namespace(
        config_file=cfg_path, mode="test",
        gpu=str(device.index) if device.type == "cuda" else "",
    ))
    model = Build_Model(cfg).to(device).eval()

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    _KEYS = ("model_state_dict", "state_dict", "model", "state", "net")
    if isinstance(ckpt, dict):
        state_dict = None
        for k in _KEYS:
            if k in ckpt:
                candidate = ckpt[k]
                if isinstance(candidate, dict) and any(kk in candidate for kk in _KEYS):
                    for kk in _KEYS:
                        if kk in candidate:
                            candidate = candidate[kk]
                            break
                state_dict = candidate
                break
        if state_dict is None:
            state_dict = ckpt
    else:
        state_dict = ckpt

    if isinstance(state_dict, dict):
        state_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}

    model.load_state_dict(state_dict, strict=True)
    metrics = Build_Metrics(cfg)
    return cfg, model, metrics


def _worker_fn(
    worker_id: int,
    cfg_path: str,
    ckpt_path: str,
    windows: List[List[Dict[str, Any]]],
    n_workers: int,
    n_warmup: int,
    result_q: mp.Queue,
    start_event: mp.Event,
    infer_mode: str = "batched",
):
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cfg, model, metrics = _load_model(cfg_path, ckpt_path, device)

        # Warm-up: run a few windows before timing starts
        warmup_windows = windows[:min(n_warmup, len(windows))]
        for w in warmup_windows:
            _predict_window(w, cfg, model, metrics, device, infer_mode)

        # Synchronise: wait until all workers are ready
        result_q.put({"status": "ready", "worker_id": worker_id})
        start_event.wait()

        # Round-robin assignment (same distribution as the real pool)
        my_windows = windows[worker_id::n_workers]

        timings = []
        t_start = time.perf_counter()

        for w in my_windows:
            t = _predict_window(w, cfg, model, metrics, device, infer_mode)
            if t["total_ms"] > 0:
                timings.append(t)

        wall_s = time.perf_counter() - t_start
        fps = len(timings) / wall_s if wall_s > 0 else 0.0

        result_q.put({
            "status": "done",
            "worker_id": worker_id,
            "n_preds": len(timings),
            "wall_s": wall_s,
            "fps": fps,
            "timings": timings,
        })

    except Exception as e:
        traceback.print_exc()
        result_q.put({"status": "error", "worker_id": worker_id, "error": str(e)})


# ──────────────────────────────────────────────────────────────────────────────
# Build sliding windows from packet list
# ──────────────────────────────────────────────────────────────────────────────
def build_windows(packets: List[Dict]) -> List[List[Dict]]:
    if len(packets) < T_O:
        raise ValueError(f"Need at least {T_O} packets, got {len(packets)}")
    return [packets[i: i + T_O] for i in range(len(packets) - T_O + 1)]


# ──────────────────────────────────────────────────────────────────────────────
# Run one benchmark configuration
# ──────────────────────────────────────────────────────────────────────────────
def run_benchmark(cfg_path, ckpt_path, windows, n_workers, run_idx, n_warmup=5, infer_mode="batched"):
    print(f"\n  [Run {run_idx+1}] workers={n_workers}, windows={len(windows)} ...")

    result_q = mp.Queue()
    start_event = mp.Event()

    procs = [
        mp.Process(
            target=_worker_fn,
            args=(wid, cfg_path, ckpt_path, windows, n_workers,
                  n_warmup, result_q, start_event, infer_mode),
            daemon=True,
        )
        for wid in range(n_workers)
    ]
    for p in procs:
        p.start()

    # Wait for all workers to finish warm-up and signal ready
    ready = 0
    while ready < n_workers:
        msg = result_q.get(timeout=300)
        if msg.get("status") == "ready":
            ready += 1
        elif msg.get("status") == "error":
            print(f"  Worker {msg['worker_id']} failed during init: {msg['error']}")

    # Fire all workers simultaneously
    t_wall_start = time.perf_counter()
    t_start_epoch = time.time()
    start_event.set()

    # Collect results
    done = 0
    results = []
    while done < n_workers:
        msg = result_q.get(timeout=3600)   # added: per-agent at 1 worker can exceed 600 s
        if msg.get("status") in ("done", "error"):
            results.append(msg)
            done += 1

    t_wall_end = time.perf_counter()
    for p in procs:
        p.join(timeout=10)

    wall_s = t_wall_end - t_wall_start

    # Aggregate
    all_timings = []
    total_preds = 0
    for r in results:
        if r.get("status") == "error":
            print(f"  Worker {r['worker_id']} error: {r.get('error')}")
            continue
        all_timings.extend(r["timings"])
        total_preds += r["n_preds"]
        if r["timings"]:
            wt = np.array([t["total_ms"] for t in r["timings"]])
            wi = np.array([t["infer_ms"] for t in r["timings"]])
            print(
                f"    P{r['worker_id']}: {r['n_preds']} preds | "
                f"FPS={r['fps']:.2f} | "
                f"infer mean={wi.mean():.1f}ms | "
                f"total mean={wt.mean():.1f}ms"
            )

    if not all_timings:
        return None

    agg_fps = total_preds / wall_s if wall_s > 0 else 0.0
    prep_arr = np.array([t["prep_ms"] for t in all_timings])
    infer_arr = np.array([t["infer_ms"] for t in all_timings])
    post_arr = np.array([t["post_ms"] for t in all_timings])
    total_arr = np.array([t["total_ms"] for t in all_timings])

    row = {
        "infer_mode": infer_mode,
        "n_agents_mean": round(float(np.mean([t["n_agents"] for t in all_timings])), 2),
        "t_start_epoch": round(t_start_epoch, 3),
        "n_workers": n_workers,
        "run": run_idx + 1,
        "n_preds": total_preds,
        "wall_s": round(wall_s, 2),
        "agg_fps": round(agg_fps, 2),
        "prep_mean_ms": round(prep_arr.mean(), 1),
        "prep_p50_ms": round(np.percentile(prep_arr, 50), 1),
        "prep_p95_ms": round(np.percentile(prep_arr, 95), 1),
        "infer_mean_ms": round(infer_arr.mean(), 1),
        "infer_p50_ms": round(np.percentile(infer_arr, 50), 1),
        "infer_p95_ms": round(np.percentile(infer_arr, 95), 1),
        "post_mean_ms": round(post_arr.mean(), 1),
        "total_mean_ms": round(total_arr.mean(), 1),
        "total_p50_ms": round(np.percentile(total_arr, 50), 1),
        "total_p95_ms": round(np.percentile(total_arr, 95), 1),
        "total_p99_ms": round(np.percentile(total_arr, 99), 1),
    }

    print(
        f"  → Aggregate: {total_preds} preds | wall={wall_s:.1f}s | "
        f"FPS={agg_fps:.2f} | "
        f"infer mean={row['infer_mean_ms']}ms | "
        f"total mean={row['total_mean_ms']}ms"
    )
    return row


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", required=True,
                    help=".pkl from generate_perception.py")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--sweep-workers", type=int, nargs="+", default=None,
                    help="e.g. --sweep-workers 1 2 4 6 8")
    ap.add_argument("--n-runs", type=int, default=3)
    ap.add_argument("--n-warmup", type=int, default=5,
                    help="Warm-up windows per worker before timing starts")
    ap.add_argument("--out", default="bench_results.csv")
    ap.add_argument("--infer-mode", choices=["batched", "per-agent"], default="batched",
                    help="per-agent: one model.predict call per agent (batch size 1), as a reference for the batched mode")
    args = ap.parse_args()

    mp.set_start_method("spawn", force=True)

    # Load packets
    print(f"Loading packets: {args.packets}")
    with open(args.packets, "rb") as f:
        packets = pickle.load(f)

    n_obj = packets[0]["perception_df"]["trackId"].nunique() if packets else 0
    n_frames = len(packets)
    print(f"  {n_frames} packets | {n_obj} objects per frame")

    windows = build_windows(packets)
    print(f"  {len(windows)} sliding windows of T_O={T_O}")

    worker_counts = args.sweep_workers or [args.n_workers]

    print(f"\n{'='*60}")
    print(f"  Prediction Benchmark  (pure predictor, no camera)")
    print(f"  Workers to test : {worker_counts}")
    print(f"  Runs per config : {args.n_runs}")
    print(f"  Warm-up windows : {args.n_warmup}")
    print(f"{'='*60}")

    all_rows = []
    for n_w in worker_counts:
        print(f"\n── {n_w} worker(s) {'─'*45}")
        run_rows = []
        for r in range(args.n_runs):
            row = run_benchmark(
                args.config, args.checkpoint, windows,
                n_w, r, n_warmup=args.n_warmup, infer_mode=args.infer_mode,
            )
            if row:
                run_rows.append(row)
                all_rows.append(row)

        valid = [r for r in run_rows if "total_mean_ms" in r]
        if valid:
            avg_fps = np.mean([r["agg_fps"] for r in valid])
            avg_lat = np.mean([r["total_mean_ms"] for r in valid])
            avg_inf = np.mean([r["infer_mean_ms"] for r in valid])
            avg_pre = np.mean([r["prep_mean_ms"] for r in valid])
            print(
                f"  ── Avg ({len(valid)} runs): "
                f"FPS={avg_fps:.2f} | "
                f"prep={avg_pre:.1f}ms | "
                f"infer={avg_inf:.1f}ms | "
                f"total={avg_lat:.1f}ms"
            )

    if not all_rows:
        print("\nNo results collected.")
        return

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(args.out, index=False)
    print(f"\nResults saved → {args.out}")

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SUMMARY (mean over runs per worker count)")
    print(f"{'='*70}")
    summary = (
        results_df.groupby("n_workers")
        .agg(
            runs=("run", "count"),
            avg_fps=("agg_fps", "mean"),
            prep_ms=("prep_mean_ms", "mean"),
            infer_ms=("infer_mean_ms", "mean"),
            post_ms=("post_mean_ms", "mean"),
            total_ms=("total_mean_ms", "mean"),
        )
        .round(2)
    )
    print(summary.to_string())
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()