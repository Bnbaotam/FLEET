#!/usr/bin/env python3
"""Runs the perception and prediction pipelines of several intersections at once and measures their throughput and latency."""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_LOGGING_STDERR_THRESHOLD", "3")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import absl.logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

import sys
import json
import time
import socket
import hashlib
import signal
import argparse
import multiprocessing as mp
from collections import deque
from queue import Empty
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from camera_processor import camera_process_video
from prediction_processor import start_prediction_pool

# ── CSV schema ────────────────────────────────────────────────────────────────

CSV_COLS = (
    ["frame_idx", "trackId", "x", "y", "cls_name", "bbox"]
    + [f"pred_x{i}" for i in range(1, 13)]
    + [f"pred_y{i}" for i in range(1, 13)]
)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _flatten_pred_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "pred_xy12" not in df.columns:
        df["pred_xy12"] = None
    for i in range(1, 13):
        df[f"pred_x{i}"] = np.nan
        df[f"pred_y{i}"] = np.nan
    for row_i, val in enumerate(df["pred_xy12"]):
        if val is None:
            continue
        try:
            arr = np.asarray(val, dtype=float)
            if arr.shape != (12, 2):
                continue
            for i in range(12):
                df.iloc[row_i, df.columns.get_loc(f"pred_x{i+1}")] = float(arr[i, 0])
                df.iloc[row_i, df.columns.get_loc(f"pred_y{i+1}")] = float(arr[i, 1])
        except Exception:
            continue
    return df


def _reorder_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in CSV_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[CSV_COLS]


def _queue_empty_safe(q) -> bool:
    try:
        return q.empty()
    except Exception:
        return False


def _fmt_q_size(q) -> str:
    try:
        return str(q.qsize())
    except Exception:
        return "?"


# ── Saver process (one per intersection) ─────────────────────────────────────

def _saver_process(
    intersection_name: str,
    q_pred: mp.Queue,
    stop: mp.Event,
    out_dir: str,
    jpg_quality: int        = 92,
    flush_every: int        = 50,
    csv_name: str           = "all_predictions.csv",
    perf_log_every: int     = 50,
    summary_q: Optional[mp.Queue] = None,
    save_frames: bool       = False,
    save_csv: bool          = True,
    reorder_wait_s: float   = 0.2,
):
    """Independent saver with ordered reorder buffer and latency tracking."""
    tag = f"[SAVER/{intersection_name}]"

    frames_dir = os.path.join(out_dir, "frames")
    if save_frames:
        _ensure_dir(frames_dir)
    if save_csv:
        _ensure_dir(out_dir)

    csv_path  = os.path.join(out_dir, csv_name)
    perf_path = os.path.join(out_dir, "perf_summary.csv")

    header_written      = save_csv and os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    perf_header_written = os.path.isfile(perf_path) and os.path.getsize(perf_path) > 0
    imwrite_params      = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]

    dup_count    = 0
    saved_frames = 0
    saver_t0     = time.perf_counter()

    batch_rows: List[pd.DataFrame] = []
    perf_rows:  List[dict]         = []
    saver_ts    = deque(maxlen=perf_log_every)

    cam_emit_min  = cam_emit_max  = None
    pred_emit_min = pred_emit_max = None
    recv_min      = recv_max      = None

    e2e_lats:   List[float] = []
    pred_lats:  List[float] = []
    saver_lats: List[float] = []

    meas_rows: List[dict] = []          # added: one row per forecast frame

    pending: Dict[int, Dict[str, Any]] = {}
    next_expected: Optional[int]       = None
    last_release_t = time.perf_counter()

    def _fps():
        if len(saver_ts) < 2:
            return 0.0
        dt = saver_ts[-1] - saver_ts[0]
        return (len(saver_ts) - 1) / dt if dt > 0 else 0.0

    def _flush():
        nonlocal header_written, perf_header_written, batch_rows, perf_rows
        if save_csv and batch_rows:
            big = pd.concat(batch_rows, ignore_index=True)
            big = big.sort_values(["frame_idx", "trackId"]).reset_index(drop=True)
            big = big.drop_duplicates(subset=["frame_idx", "trackId"], keep="first")
            big = _reorder_cols(big)
            big.to_csv(csv_path, index=False, mode="a", header=not header_written)
            header_written = True
            batch_rows = []
        if perf_rows:
            _ensure_dir(out_dir)
            pd.DataFrame(perf_rows).to_csv(
                perf_path, index=False, mode="a", header=not perf_header_written
            )
            perf_header_written = True
            perf_rows.clear()

    def _process(pkt: Dict[str, Any]):
        nonlocal dup_count, saved_frames
        nonlocal cam_emit_min, cam_emit_max, pred_emit_min, pred_emit_max
        nonlocal recv_min, recv_max

        t_recv = time.perf_counter()
        df     = pkt.get("perception_df", None)
        fidx   = int(pkt.get("frame_idx", -1))
        frame  = pkt.get("frame_bgr", None)
        if df is None or fidx < 0:
            return

        cam_t  = pkt.get("cam_t_emit",  None)
        pred_t = pkt.get("pred_t_emit", None)

        if cam_t  is not None:
            cam_emit_min  = cam_t  if cam_emit_min  is None else min(cam_emit_min,  cam_t)
            cam_emit_max  = cam_t  if cam_emit_max  is None else max(cam_emit_max,  cam_t)
        if pred_t is not None:
            pred_emit_min = pred_t if pred_emit_min is None else min(pred_emit_min, pred_t)
            pred_emit_max = pred_t if pred_emit_max is None else max(pred_emit_max, pred_t)
        recv_min = t_recv if recv_min is None else min(recv_min, t_recv)
        recv_max = t_recv if recv_max is None else max(recv_max, t_recv)

        e2e_ms   = (t_recv - cam_t)  * 1e3 if cam_t  is not None else None
        pred_ms  = (pred_t - cam_t)  * 1e3 if (cam_t is not None and pred_t is not None) else None
        saver_ms = (t_recv - pred_t) * 1e3 if pred_t is not None else None

        perf_rows.append({
            "frame_idx":      fidx,
            "e2e_ms":         round(e2e_ms,   2) if e2e_ms   is not None else None,
            "pred_ms":        round(pred_ms,  2) if pred_ms  is not None else None,
            "saver_queue_ms": round(saver_ms, 2) if saver_ms is not None else None,
        })
        if pkt.get("meas") is not None:     # added
            meas_rows.append({"frame_idx": fidx, **pkt["meas"], "t_recv": t_recv})
        if e2e_ms   is not None: e2e_lats.append(e2e_ms)
        if pred_ms  is not None: pred_lats.append(pred_ms)
        if saver_ms is not None: saver_lats.append(saver_ms)

        if save_frames and frame is not None:
            try:
                cv2.imwrite(
                    os.path.join(frames_dir, f"{fidx:06d}.jpg"),
                    frame, imwrite_params,
                )
            except Exception as e:
                print(f"{tag} Failed to write frame {fidx}: {e}")

        if save_csv and isinstance(df, pd.DataFrame) and not df.empty:
            try:
                dfs = _flatten_pred_cols(df)
                dfs["frame_idx"] = fidx
                if "bbox" in dfs.columns:
                    dfs["bbox"] = dfs["bbox"].map(
                        lambda b: json.dumps(list(b))
                        if b is not None and not isinstance(b, str) else b
                    )
                dfs = dfs.drop_duplicates(subset=["frame_idx", "trackId"], keep="last")
                dfs = _reorder_cols(dfs)
                batch_rows.append(dfs)
            except Exception as e:
                print(f"{tag} CSV append failed frame {fidx}: {e}")

        saved_frames += 1
        saver_ts.append(t_recv)

        if saved_frames % flush_every == 0:
            _flush()
            fps = saved_frames / max(1e-9, time.perf_counter() - saver_t0)
            print(f"{tag} {saved_frames} frames  dup_dropped={dup_count}  {fps:.2f} FPS")

            if saved_frames % perf_log_every == 0 and e2e_lats:
                e2a = np.array(e2e_lats)
                pa  = np.array(pred_lats)  if pred_lats  else np.array([np.nan])
                sa  = np.array(saver_lats) if saver_lats else np.array([np.nan])
                print(
                    f"{tag} E2E mean={np.nanmean(e2a):.1f}ms "
                    f"p50={np.nanpercentile(e2a,50):.1f}ms "
                    f"p95={np.nanpercentile(e2a,95):.1f}ms  "
                    f"cam→pred={np.nanmean(pa):.1f}ms  "
                    f"pred→saver={np.nanmean(sa):.1f}ms"
                )
                e2e_lats.clear(); pred_lats.clear(); saver_lats.clear()

    def _release(force_gap: bool = False):
        nonlocal next_expected, last_release_t
        if not pending:
            return
        if next_expected is None:
            next_expected = min(pending.keys())
        if force_gap and next_expected not in pending:
            next_expected = min(pending.keys())
        while next_expected in pending:
            _process(pending.pop(next_expected))
            last_release_t = time.perf_counter()
            next_expected += 1

    try:
        while True:
            got = False
            try:
                pkt = q_pred.get(timeout=0.05)
                got = True
            except Empty:
                pkt = None

            if got and pkt is not None:
                fidx = int(pkt.get("frame_idx", -1))
                if fidx >= 0:
                    if next_expected is not None and fidx < next_expected:
                        _process(pkt)       # fix: a late frame used to wait in the buffer until a gap appeared
                    elif fidx not in pending:
                        pending[fidx] = pkt
                    else:
                        dup_count += 1
                _release(force_gap=False)

            if pending and (time.perf_counter() - last_release_t) >= reorder_wait_s:
                _release(force_gap=True)

            if stop.is_set() and not got:
                break

        while pending:
            _release(force_gap=True)

    finally:
        _flush()
        if meas_rows:                        # added
            pd.DataFrame(meas_rows).to_csv(os.path.join(out_dir, "frames.csv"), index=False)
        with open(os.path.join(out_dir, "saver.json"), "w") as f:
            json.dump({"saved_frames": saved_frames, "dup_dropped": dup_count}, f)

    # Compute per-module FPS
    cam_fps  = pred_fps  = saver_fps = 0.0
    if saved_frames >= 2:
        if cam_emit_min is not None and cam_emit_max > cam_emit_min:
            cam_fps   = (saved_frames - 1) / (cam_emit_max  - cam_emit_min)
        if pred_emit_min is not None and pred_emit_max > pred_emit_min:
            pred_fps  = (saved_frames - 1) / (pred_emit_max - pred_emit_min)
        if recv_min is not None and recv_max > recv_min:
            saver_fps = (saved_frames - 1) / (recv_max - recv_min)

    # Full latency stats from persisted perf CSV
    lat: Dict[str, Any] = {}
    if os.path.isfile(perf_path):
        try:
            pf = pd.read_csv(perf_path)
            for col, key in [
                ("e2e_ms",         "e2e"),
                ("pred_ms",        "cam_to_pred"),
                ("saver_queue_ms", "pred_to_saver"),
            ]:
                arr = pf[col].dropna().to_numpy(dtype=float) if col in pf.columns else np.array([])
                lat[f"{key}_mean_ms"] = float(np.nanmean(arr))          if arr.size else None
                lat[f"{key}_p50_ms"]  = float(np.nanpercentile(arr, 50)) if arr.size else None
                lat[f"{key}_p95_ms"]  = float(np.nanpercentile(arr, 95)) if arr.size else None
        except Exception:
            pass

    summary = {
        "saved_frames":       saved_frames,
        "dup_dropped":        dup_count,
        "camera_avg_fps":     round(cam_fps,   3),
        "prediction_avg_fps": round(pred_fps,  3),
        "saver_avg_fps":      round(saver_fps, 3),
        "csv_path":           csv_path if save_csv else None,
        "perf_path":          perf_path,
        **lat,
    }
    if summary_q is not None:
        try:
            summary_q.put_nowait(summary)
        except Exception:
            pass

    fps_total = saved_frames / max(1e-9, time.perf_counter() - saver_t0)
    print(
        f"{tag} Done. saved={saved_frames}  dup_dropped={dup_count}  "
        f"total_throughput={fps_total:.2f} FPS"
    )


# ── One intersection: spawn all processes, return handles ────────────────────

class IntersectionPipeline:
    """All processes for one intersection."""

    def __init__(
        self,
        name:        str,
        video:       str,
        out_dir:     str,
        yolo_model:  str,
        config:      str,
        checkpoint:  str,
        target_fps:  float = 10.0,
        cam_device:  str   = "0",
        pred_device: str   = "cuda:0",
        imgsz:       int   = 640,
        half:        bool  = False,
        workers:     int   = 4,
        jpg_quality: int   = 92,
        csv_name:    str   = "all_predictions.csv",
        csv_flush_every: int   = 50,
        perf_log_every:  int   = 50,
        reorder_wait:    float = 0.2,
        save_frames: bool  = False,
        save_csv:    bool  = True,
        pace:        str   = "original",
        tracker:     Optional[str]   = None,
        start_s:     float = 0.0,
        duration_s:  Optional[float] = None,
        ready_q:     Optional[mp.Queue] = None,
        go_event:    Optional[mp.Event] = None,
    ):
        self.name     = name
        self.video    = video
        self.out_dir  = out_dir
        self.wall_start: Optional[float] = None
        self.wall_end:   Optional[float] = None

        _ensure_dir(out_dir)

        self.q_phi      = mp.Queue(maxsize=1024)
        self.q_snapshot = mp.Queue(maxsize=512)
        self.q_pred     = mp.Queue(maxsize=1024)
        self.q_summary  = mp.Queue(maxsize=2)

        self.stop_camera = mp.Event()
        self.stop_saver  = mp.Event()

        # Camera
        self.cam_proc = mp.Process(
            target=camera_process_video,
            args=(video, yolo_model, self.q_phi, self.stop_camera,
                  target_fps, cam_device, imgsz, half, save_frames),
            kwargs=dict(pace=pace, tracker=tracker, start_s=start_s, duration_s=duration_s,
                        out_dir=out_dir, ready_q=ready_q, go_event=go_event, site=name),
            daemon=True,
            name=f"cam-{name}",
        )

        # Prediction pool (feeder + workers)
        self.pred_procs, self.stop_pool = start_prediction_pool(
            cfg_path=config,
            ckpt_path=checkpoint,
            q_phi=self.q_phi,
            q_snapshot=self.q_snapshot,
            q_pred=self.q_pred,
            shared=None,
            n_workers=workers,
            pred_device=pred_device,
            save_frames=save_frames,
            perf_log_every=perf_log_every,
            ready_q=ready_q,
            site=name,
            out_dir=out_dir,
        )

        # Saver
        self.saver_proc = mp.Process(
            target=_saver_process,
            args=(name, self.q_pred, self.stop_saver, out_dir,
                  jpg_quality, csv_flush_every, csv_name,
                  perf_log_every, self.q_summary,
                  save_frames, save_csv, reorder_wait),
            daemon=True,
            name=f"saver-{name}",
        )

    def start(self):
        self.wall_start = time.perf_counter()
        self.cam_proc.start()
        self.saver_proc.start()
        # pred_procs already started by start_prediction_pool
        print(
            f"[MULTI] Started  {self.name:12s}  "
            f"cam={self.cam_proc.pid}  "
            f"saver={self.saver_proc.pid}  "
            f"pred_workers={len(self.pred_procs)-1}"
        )

    def camera_alive(self) -> bool:
        return self.cam_proc.is_alive()

    def is_done(self) -> bool:
        """Returns True once camera is dead AND saver is dead."""
        return not self.cam_proc.is_alive() and not self.saver_proc.is_alive()

    def drain_and_stop(self):
        """Signal stop chain and wait for orderly shutdown."""
        # 1. Stop camera
        self.stop_camera.set()
        self.cam_proc.join(timeout=3.0)
        if self.cam_proc.is_alive():
            self.cam_proc.terminate()

        # 2. Drain phi + snapshot queues, then stop predictor pool
        _wait_until_queues_drain(
            [self.q_phi, self.q_snapshot],
            stable_seconds=1.0, poll_s=0.25,
        )
        self.stop_pool.set()
        for p in self.pred_procs:
            p.join(timeout=5.0)
        for p in self.pred_procs:
            if p.is_alive():
                p.terminate()

        # 3. Drain pred queue, then stop saver
        _wait_until_queues_drain(
            [self.q_pred],
            stable_seconds=1.0, poll_s=0.25,
        )
        self.stop_saver.set()
        self.saver_proc.join(timeout=20.0)
        if self.saver_proc.is_alive():
            self.saver_proc.terminate()

        self.wall_end = time.perf_counter()

    def collect_summary(self) -> Dict[str, Any]:
        summary: Optional[Dict[str, Any]] = None
        try:
            while True:
                summary = self.q_summary.get_nowait()
        except Empty:
            pass
        if summary is None:
            summary = {}
        summary["intersection"] = self.name
        summary["video"]        = self.video
        wall = (self.wall_end - self.wall_start) if self.wall_end else 0.0
        summary["wall_time_s"]  = round(wall, 2)
        return summary


# ── Queue drain (plain function, no process) ─────────────────────────────────

def _wait_until_queues_drain(queues, stable_seconds=1.0, poll_s=0.25, label=""):
    stable_since = None
    while True:
        all_empty = all(_queue_empty_safe(q) for q in queues)
        if all_empty:
            if stable_since is None:
                stable_since = time.perf_counter()
            elif time.perf_counter() - stable_since >= stable_seconds:
                break
        else:
            stable_since = None
        time.sleep(poll_s)


# ── Performance report ────────────────────────────────────────────────────────

def _print_report(results: List[Dict[str, Any]], report_path: str):
    bar = "=" * 100
    print(f"\n{bar}")
    print("  MULTI-INTERSECTION PARALLEL SCALING REPORT")
    print(bar)

    rows = []
    for r in results:
        name = r.get("intersection", "?")
        print(f"\n  [{name}]  video = {r.get('video', '?')}")
        print(f"    Wall time            : {r.get('wall_time_s', '?')} s")
        print(f"    Saved frames         : {r.get('saved_frames', '?')}")
        print(f"    Dup-dropped          : {r.get('dup_dropped', '?')}")
        print(f"    ── Module FPS ──────────────────────────────")
        print(f"    Camera Perception    : {r.get('camera_avg_fps', 0):.2f} FPS")
        print(f"    Trajectory Prediction: {r.get('prediction_avg_fps', 0):.2f} FPS")
        print(f"    Saver                : {r.get('saver_avg_fps', 0):.2f} FPS")
        print(f"    ── Latency (ms) ────────────────────────────")
        print(f"    E2E   mean/p50/p95   : "
              f"{r.get('e2e_mean_ms','N/A')} / "
              f"{r.get('e2e_p50_ms','N/A')} / "
              f"{r.get('e2e_p95_ms','N/A')}")
        print(f"    Cam → Pred mean      : {r.get('cam_to_pred_mean_ms','N/A')} ms")
        print(f"    Pred → Saver mean    : {r.get('pred_to_saver_mean_ms','N/A')} ms")
        print(f"    ── Output paths ────────────────────────────")
        print(f"    CSV                  : {r.get('csv_path','—')}")
        print(f"    Per-frame perf CSV   : {r.get('perf_path','—')}")

        rows.append({
            "intersection":            name,
            "video":                   r.get("video", ""),
            "wall_time_s":             r.get("wall_time_s", None),
            "saved_frames":            r.get("saved_frames", None),
            "dup_dropped":             r.get("dup_dropped", None),
            "camera_avg_fps":          round(r.get("camera_avg_fps",    0), 3),
            "prediction_avg_fps":      round(r.get("prediction_avg_fps",0), 3),
            "saver_avg_fps":           round(r.get("saver_avg_fps",     0), 3),
            "e2e_mean_ms":             r.get("e2e_mean_ms",            None),
            "e2e_p50_ms":              r.get("e2e_p50_ms",             None),
            "e2e_p95_ms":              r.get("e2e_p95_ms",             None),
            "cam_to_pred_mean_ms":     r.get("cam_to_pred_mean_ms",    None),
            "cam_to_pred_p50_ms":      r.get("cam_to_pred_p50_ms",     None),
            "cam_to_pred_p95_ms":      r.get("cam_to_pred_p95_ms",     None),
            "pred_to_saver_mean_ms":   r.get("pred_to_saver_mean_ms",  None),
            "pred_to_saver_p50_ms":    r.get("pred_to_saver_p50_ms",   None),
            "pred_to_saver_p95_ms":    r.get("pred_to_saver_p95_ms",   None),
            "csv_path":                r.get("csv_path", ""),
            "perf_path":               r.get("perf_path", ""),
        })

    n = len(results)
    if n > 1:
        print(f"\n{bar}")
        print(f"  AGGREGATE  ({n} intersections in parallel)")
        print(f"{bar}")
        cam_fps_vals  = [r.get("camera_avg_fps",    0) for r in results]
        pred_fps_vals = [r.get("prediction_avg_fps",0) for r in results]
        sav_fps_vals  = [r.get("saver_avg_fps",     0) for r in results]
        e2e_vals      = [r.get("e2e_mean_ms") for r in results if r.get("e2e_mean_ms") is not None]
        print(f"  Total camera FPS across all intersections : {sum(cam_fps_vals):.2f}")
        print(f"  Mean  camera FPS per intersection         : {np.mean(cam_fps_vals):.2f}")
        print(f"  Total pred   FPS across all intersections : {sum(pred_fps_vals):.2f}")
        print(f"  Mean  pred   FPS per intersection         : {np.mean(pred_fps_vals):.2f}")
        print(f"  Total saver  FPS across all intersections : {sum(sav_fps_vals):.2f}")
        print(f"  Mean  E2E latency (all intersections)     : "
              f"{np.mean(e2e_vals):.1f} ms" if e2e_vals else "  Mean E2E latency: N/A")

    _ensure_dir(str(Path(report_path).parent))
    pd.DataFrame(rows).to_csv(report_path, index=False)
    print(f"\n  Report → {report_path}\n{bar}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Run N intersections SIMULTANEOUSLY as parallel pipeline instances "
            "and measure per-module + end-to-end performance for each."
        )
    )

    # Input
    ap.add_argument("--videos",      nargs="+", required=True,
                    help="List of input video paths (one per intersection)")
    ap.add_argument("--names",       nargs="+", default=None,
                    help="Intersection names. Default: INT-0, INT-1, ...")
    ap.add_argument("--yolo-model",  required=True, help="YOLO model weights")

    # Pipeline
    ap.add_argument("--config",      required=True, help="FlowChain config file")
    ap.add_argument("--checkpoint",  required=True, help="FlowChain checkpoint")
    ap.add_argument("--workers",     type=int,   default=4,
                    help="Prediction workers per intersection (default 4)")
    ap.add_argument("--target-fps",  type=float, default=10.0,
                    help="Sample FPS from each video (default 10)")
    ap.add_argument("--cam-device",  default="0",
                    help="YOLO device for all cameras (default 0 = cuda:0)")
    ap.add_argument("--pred-device", default="cuda:0",
                    help="FlowChain device for all predictors (default cuda:0)")
    ap.add_argument("--imgsz",       type=int,   default=640)
    ap.add_argument("--half",        action="store_true", help="Not used: the flag is not passed to YOLO (an engine's precision is fixed at export)")

    # Output
    ap.add_argument("--out",         default="results",
                    help="Root output dir. Each intersection → results/<name>/")
    ap.add_argument("--jpg-quality", type=int,   default=92)
    ap.add_argument("--csv-name",    default="all_predictions.csv")
    ap.add_argument("--csv-flush-every",  type=int,   default=50)
    ap.add_argument("--perf-log-every",   type=int,   default=50)
    ap.add_argument("--reorder-wait",     type=float, default=0.2,
                    help="Saver reorder buffer max wait (default 0.2 s)")

    # Save modes
    ap.add_argument("--no-save",         action="store_true",
                    help="Write neither the predictions CSV nor JPEG frames (the timing files are still written)")
    ap.add_argument("--save-csv-only",   action="store_true",
                    help="Write CSV only, skip JPEG frames (fastest I/O)")
    ap.add_argument("--save-video-only", action="store_true",
                    help="Write JPEG frames only, skip CSV")

    # measurement options (defaults = original behaviour)
    ap.add_argument("--pace", choices=["original", "arrival", "none"], default="original",
                    help="original: as before; arrival: release each frame at its live-camera time; none: capacity")
    ap.add_argument("--tracker", default=None,
                    help="Ultralytics tracker yaml, e.g. bytetrack.yaml (default: library default, as before)")
    ap.add_argument("--start-s", type=float, nargs="+", default=None,
                    help="Start offset in seconds, one per video (default 0)")
    ap.add_argument("--duration-s", type=float, default=None,
                    help="Seconds of video per intersection (default: to the end)")
    ap.add_argument("--sync-start", action="store_true",
                    help="Start all streams together once every camera and worker is ready")

    args = ap.parse_args()
    mp.set_start_method("spawn", force=True)

    modes = int(args.no_save) + int(args.save_csv_only) + int(args.save_video_only)
    if modes > 1:
        ap.error("Choose at most one of --no-save, --save-csv-only, --save-video-only")
    if args.no_save:
        save_frames, save_csv = False, False
    elif args.save_video_only:
        save_frames, save_csv = True, False
    elif args.save_csv_only:
        save_frames, save_csv = False, True
    else:
        save_frames, save_csv = True, True

    videos = args.videos
    names  = args.names or [f"INT-{i}" for i in range(len(videos))]
    if len(names) != len(videos):
        ap.error(f"--names ({len(names)}) must match --videos ({len(videos)})")
    starts = args.start_s or [0.0] * len(videos)
    if len(starts) != len(videos):
        ap.error(f"--start-s ({len(starts)}) must match --videos ({len(videos)})")

    # added: the patched modules next to this file must be the ones imported
    import camera_processor, prediction_processor
    for m in (camera_processor, prediction_processor):
        assert os.path.dirname(os.path.abspath(m.__file__)) == ROOT, f"wrong module: {m.__file__}"
    _ensure_dir(args.out)
    run_meta: Dict[str, Any] = {"t_launch": time.perf_counter(), "epoch_launch": time.time()}
    run_meta["epoch_minus_perf_launch"] = run_meta["epoch_launch"] - run_meta["t_launch"]
    _write_run_config(args, names, videos, starts)
    ready_q  = mp.Queue() if args.sync_start else None
    go_event = mp.Event() if args.sync_start else None

    # ── Build pipeline instances ──────────────────────────────────────────
    pipelines: List[IntersectionPipeline] = []
    for name, video, start_s in zip(names, videos, starts):
        out_dir = os.path.join(args.out, name.replace(" ", "_"))
        pip = IntersectionPipeline(
            name=name,
            video=video,
            out_dir=out_dir,
            yolo_model=args.yolo_model,
            config=args.config,
            checkpoint=args.checkpoint,
            target_fps=args.target_fps,
            cam_device=args.cam_device,
            pred_device=args.pred_device,
            imgsz=args.imgsz,
            half=args.half,
            workers=args.workers,
            jpg_quality=args.jpg_quality,
            csv_name=args.csv_name,
            csv_flush_every=args.csv_flush_every,
            perf_log_every=args.perf_log_every,
            reorder_wait=args.reorder_wait,
            save_frames=save_frames,
            save_csv=save_csv,
            pace=args.pace,
            tracker=args.tracker,
            start_s=start_s,
            duration_s=args.duration_s,
            ready_q=ready_q,
            go_event=go_event,
        )
        pipelines.append(pip)

    # ── Start ALL intersections simultaneously ────────────────────────────
    global_wall_start = time.perf_counter()
    print(f"\n[MULTI] Launching {len(pipelines)} intersections in PARALLEL...\n")
    for pip in pipelines:
        pip.start()
    print(f"\n[MULTI] All {len(pipelines)} intersections are running.\n")

    if args.sync_start:                      # added: start every stream together
        expected = len(pipelines) * (1 + args.workers)
        ready, deadline = [], time.perf_counter() + 900
        while len(ready) < expected:
            try:
                ready.append(ready_q.get(timeout=1.0))
            except Empty:
                pass
            dead = [f"{pip.name}/{p.name}" for pip in pipelines
                    for p in [pip.cam_proc] + pip.pred_procs if not p.is_alive()]
            if dead:
                raise RuntimeError(f"[MULTI] process died before it was ready: {dead}")
            if time.perf_counter() > deadline:
                raise TimeoutError(f"[MULTI] only {len(ready)}/{expected} processes ready after 900 s")
        run_meta["ready"] = ready
        run_meta["t_go"], run_meta["epoch_go"] = time.perf_counter(), time.time()
        go_event.set()
        print(f"[MULTI] All {expected} camera/worker processes ready after "
              f"{run_meta['t_go'] - run_meta['t_launch']:.1f} s. Streams started.")

    # ── Graceful shutdown handler ─────────────────────────────────────────
    forced_stop = mp.Event()

    def _handle_sig(signum, frame):
        forced_stop.set()
        print(f"\n[MULTI] Signal {signum} received — stopping all intersections...")

    signal.signal(signal.SIGINT,  _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    # ── Monitor: drain each intersection in a background thread ─────────────
    import threading

    # Grace period: wait until all cameras are confirmed alive before monitoring
    print("[MULTI] Waiting for all cameras to start...")
    for pip in pipelines:
        for _ in range(20):            # up to 10s per camera
            if pip.camera_alive():
                break
            time.sleep(0.5)
        else:
            print(f"[MULTI] WARNING: {pip.name} camera never started!")

    pending       = list(pipelines)
    drain_threads: List[threading.Thread] = []
    draining_set: set = set()

    def _drain_one(pip: "IntersectionPipeline"):
        try:
            pip.drain_and_stop()
        except Exception as e:
            print(f"[MULTI] drain_and_stop error for {pip.name}: {e}")
        print(f"[MULTI] {pip.name} fully finished.")

    try:
        while pending or any(t.is_alive() for t in drain_threads):
            time.sleep(0.5)

            if forced_stop.is_set():
                for pip in pipelines:
                    pip.stop_camera.set()
                    pip.stop_pool.set()
                    pip.stop_saver.set()
                break

            still_pending = []
            for pip in pending:
                if not pip.camera_alive():
                    if pip.name not in draining_set:
                        print(f"[MULTI] Camera done for {pip.name} — draining in background...")
                        draining_set.add(pip.name)
                        t = threading.Thread(
                            target=_drain_one, args=(pip,),
                            daemon=False,          # ← non-daemon: keeps main alive
                        )
                        drain_threads.append(t)
                        t.start()
                else:
                    still_pending.append(pip)
            pending = still_pending

    except KeyboardInterrupt:
        forced_stop.set()
        for pip in pipelines:
            pip.stop_camera.set()
            pip.stop_pool.set()
            pip.stop_saver.set()

    # Wait for all drain threads
    for t in drain_threads:
        t.join(timeout=60.0)

    # ── Orderly teardown for any remaining pipelines ──────────────────────
    for pip in pipelines:
        if not pip.is_done():
            pip.drain_and_stop()
        elif pip.wall_end is None:
            pip.wall_end = time.perf_counter()

    global_wall = time.perf_counter() - global_wall_start
    print(f"\n[MULTI] All intersections done.  Global wall time: {global_wall:.2f} s")

    # ── Collect and report ────────────────────────────────────────────────
    results = [pip.collect_summary() for pip in pipelines]
    report_path = os.path.join(args.out, "performance_report.csv")
    _print_report(results, report_path)

    # added: run timing and process exit codes
    run_meta.update(t_end=time.perf_counter(), epoch_end=time.time(), global_wall_s=global_wall,
                    sites={pip.name: {"wall_start": pip.wall_start, "wall_end": pip.wall_end} for pip in pipelines},
                    exit_codes={f"{pip.name}/{p.name}": p.exitcode for pip in pipelines
                                for p in [pip.cam_proc, pip.saver_proc] + pip.pred_procs})
    run_meta["epoch_minus_perf_end"] = run_meta["epoch_end"] - run_meta["t_end"]
    with open(os.path.join(args.out, "run_meta.json"), "w") as f:
        json.dump(run_meta, f, indent=1, default=float)
    bad = {k: v for k, v in run_meta["exit_codes"].items() if v != 0}
    if args.sync_start and bad:
        raise SystemExit(f"[MULTI] processes failed: {bad}")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _write_run_config(args, names, videos, starts):
    """Added: every input and option that changes behaviour, next to the outputs."""
    import torch, ultralytics
    from ultralytics.cfg import get_cfg
    vids = []
    for name, v, s in zip(names, videos, starts):
        cap = cv2.VideoCapture(v)
        assert cap.isOpened(), f"cannot open {v}"
        fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        stride = max(1, round(fps / args.target_fps))
        vids.append({"site": name, "path": os.path.abspath(v), "sha256": _sha256(v), "fps": fps,
                     "frames": n, "length_s": n / fps, "start_s": s, "duration_s": args.duration_s,
                     "stride": stride, "effective_hz": fps / stride})
    cfg = {
        "argv": sys.argv, "args": vars(args), "cwd": os.getcwd(), "host": socket.gethostname(),
        "tracker_used": args.tracker or f"library default ({get_cfg().tracker})",
        "ultralytics_default_tracker": get_cfg().tracker,
        "engine": {"path": os.path.abspath(args.yolo_model), "sha256": _sha256(args.yolo_model)},
        "checkpoint": {"path": os.path.abspath(args.checkpoint), "sha256": _sha256(args.checkpoint)},
        "config": {"path": os.path.abspath(args.config), "sha256": _sha256(args.config),
                   "text": open(args.config).read()},
        "videos": vids,
        "code_sha256": {f: _sha256(os.path.join(ROOT, f)) for f in sorted(os.listdir(ROOT)) if f.endswith(".py")},
        "versions": {"python": sys.version, "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                     "opencv": cv2.__version__},
        "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "PYTHONPATH")},
    }
    with open(os.path.join(args.out, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=1, default=str)


if __name__ == "__main__":
    main()
