#!/usr/bin/env python3
"""
camera_processor.py

Emits packets to cam_queue:
  "perception_df" : pd.DataFrame | None   (ONLY classes 0,1,2,3,5,7)
  "frame_idx"     : int
  "cam_t_emit"    : float
  "frame_bgr"     : np.ndarray            (ONLY when save_frames=True)

Filtered COCO classes kept:
  0 person | 1 bicycle | 2 car | 3 motorcycle | 5 bus | 7 truck
All other detections are silently dropped before the packet is emitted.

Measurement-only additions for the Jetson AGX Thor runs (see the docstring of camera_process_video).
Defaults keep the original behaviour.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_LOGGING_STDERR_THRESHOLD", "3")
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import sys, cv2, math, time, threading
import multiprocessing as mp
from collections import deque
from queue import Empty, Full, Queue as TQueue
from typing import Optional

import json
import numpy as np
import pandas as pd
from ultralytics import YOLO

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

from common import make_df_from_results

# Classes we care about (COCO IDs)
KEEP_CLASSES = {0, 1, 2, 3, 5, 7}
KEEP_CLASS_NAMES = {
    0: "person", 1: "bicycle", 2: "car",
    3: "motorcycle", 5: "bus", 7: "truck",
}


class _VideoReader:
    """Background-thread ring-buffer video reader."""
    def __init__(self, path: str, maxsize: int = 32, skip: int = 0):
        self._cap    = cv2.VideoCapture(path)
        self._q      = TQueue(maxsize=maxsize)
        self._stop   = threading.Event()
        self._skip   = skip                   # added: frames to skip before the first read
        self.skipped = threading.Event()      # added: set once the skip is done
        self._opened = self._cap.isOpened()
        self._t      = threading.Thread(target=self._run, daemon=True)
        if self._opened:
            self._t.start()

    def _run(self):
        for _ in range(self._skip):           # added: start offset
            if not self._cap.grab():
                break
        self.skipped.set()
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                self._q.put((False, None))
                break
            self._q.put((True, frame))
        self._cap.release()

    def isOpened(self): return self._opened
    def read(self):     return self._q.get()
    def get(self, p):   return self._cap.get(p)

    def release(self):
        self._stop.set()
        while not self._q.empty():
            try: self._q.get_nowait()
            except: break
        self._t.join(timeout=2.0)


class _RollingFPS:
    def __init__(self, window: int = 60):
        self._ts = deque(maxlen=window)

    def tick(self) -> float:
        self._ts.append(time.perf_counter())
        if len(self._ts) < 2: return 0.0
        dt = self._ts[-1] - self._ts[0]
        return (len(self._ts) - 1) / dt if dt > 0 else 0.0


def _filter_classes(df):
    """Drop rows whose cls_id is not in KEEP_CLASSES. Returns None if empty."""
    if df is None:
        return None
    if "cls_id" in df.columns:
        df = df[df["cls_id"].isin(KEEP_CLASSES)].copy()
    elif "cls_name" in df.columns:
        df = df[df["cls_name"].isin(KEEP_CLASS_NAMES.values())].copy()
    return df if not df.empty else None


def camera_process_video(
    video_path:      str,
    yolo_model_path: str,
    cam_queue:       mp.Queue,
    stop_event:      mp.Event,
    target_fps:      float = 30.0,
    device:          str   = "0",
    imgsz:           int   = 640,
    half:            bool  = False,
    save_frames:     bool  = False,
    summary_q:       Optional[mp.Queue] = None,
    pace:            str   = "original",
    tracker:         Optional[str] = None,
    start_s:         float = 0.0,
    duration_s:      Optional[float] = None,
    out_dir:         Optional[str] = None,
    ready_q:         Optional[mp.Queue] = None,
    go_event:        Optional[mp.Event] = None,
    site:            str   = "",
):
    """
    YOLO detect+track -> filter to KEEP_CLASSES -> emit lightweight packets.

    save_frames=False (default):
        packet = {perception_df, frame_idx, cam_t_emit}           ~KB
    save_frames=True:
        packet = {perception_df, frame_idx, cam_t_emit, frame_bgr}  ~3 MB

    Never pass save_frames=True unless JPEG output is needed; including
    frame_bgr serialises ~83 MB/s through IPC and will starve prediction workers.

    Measurement additions (the defaults keep the original behaviour):
      pace="original": original scheduler (sleep to the next slot; when late, jump ahead; frames are never skipped)
      pace="arrival" : frame k is released at its live-camera arrival time t_stream0 + k_src/src_fps;
                       when perception is late, the next frame is processed at once (frames are never skipped)
      pace="none"    : no sleep (capacity)
      tracker        : Ultralytics tracker yaml, e.g. "bytetrack.yaml"; None = library default (as before)
      start_s, duration_s : read only video[start_s, start_s + duration_s)
      ready_q, go_event   : report ready after model load and warm-up, then wait for the common start
      out_dir        : write camera_frames.csv and camera.json there
    Every packet carries "meas": per-frame time.perf_counter() stamps (CLOCK_MONOTONIC, comparable across processes).
    """
    assert pace in ("original", "arrival", "none"), pace
    track_kw = {"tracker": tracker} if tracker else {}
    rows: list = []
    q_drops: list = []
    info: dict = {"site": site, "video": video_path, "pace": pace, "tracker": tracker,
                  "start_s": start_s, "duration_s": duration_s, "target_fps": target_fps}
    emitted_idx = 0
    try:
        model = YOLO(yolo_model_path, task="detect")

        probe = cv2.VideoCapture(video_path)
        if not probe.isOpened():
            raise IOError(f"[CAM] cannot open {video_path}")
        src_fps = probe.get(cv2.CAP_PROP_FPS)
        n_video = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
        probe.release()
        if not isinstance(src_fps, (int, float)) or src_fps <= 0 or math.isnan(src_fps):
            src_fps = 30.0
        sample_stride = max(1, round(src_fps / float(target_fps)))
        start_frame = int(round(start_s * src_fps))
        n_read = None if duration_s is None else int(round(duration_s * src_fps))
        if pace != "original":
            assert abs(src_fps / sample_stride - target_fps) < 0.01, \
                f"{video_path}: {src_fps} FPS / stride {sample_stride} is not {target_fps} Hz"
        if n_read is not None:
            assert start_frame + n_read <= n_video, \
                f"{video_path}: needs frames up to {start_frame + n_read}, the file has {n_video}"
        info.update(src_fps=src_fps, stride=sample_stride, video_frames=n_video,
                    start_frame=start_frame, n_read=n_read)

        cap = _VideoReader(video_path, maxsize=32, skip=start_frame)
        if not cap.isOpened():
            raise IOError(f"[CAM] cannot open {video_path}")
        print(f"[CAM] src_fps={src_fps:.1f}  stride={sample_stride}  "
              f"target={target_fps:.1f} FPS  save_frames={save_frames}  "
              f"keep_classes={sorted(KEEP_CLASSES)}  pace={pace}  tracker={tracker or 'library default'}  "
              f"start_frame={start_frame}  n_read={n_read}")

        # TRT / model warm-up (added: with the run's tracker, because persist=True keeps the warm-up tracker)
        _dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        for _ in range(3):
            model.track(_dummy, device=device, imgsz=imgsz, persist=False,
                        verbose=False, classes=list(KEEP_CLASSES), **track_kw)
        print("[CAM] Warm-up done.")
        cap.skipped.wait()

        if ready_q is not None:                      # added: common start
            info["t_ready"] = time.perf_counter()
            ready_q.put({"proc": "camera", "site": site, "t": info["t_ready"]})
            go_event.wait()
        t_stream0 = time.perf_counter()              # a live camera starts delivering here
        info["t_stream0"] = t_stream0

        fps_tracker  = _RollingFPS(window=60)
        det_lats: list = []
        frame_period   = 1.0 / float(target_fps)
        next_due       = time.perf_counter()
        src_idx = emitted_idx = frame_count = 0
        t_first = t_last = None

        while not stop_event.is_set():
            if n_read is not None and src_idx >= n_read:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if (src_idx % sample_stride) != 0:
                src_idx += 1
                continue

            t_arrive = t_stream0 + src_idx / src_fps     # when a live camera would deliver this frame
            if pace == "arrival":
                now = time.perf_counter()
                if now < t_arrive:
                    time.sleep(t_arrive - now)

            t0      = time.perf_counter()
            results = model.track(
                frame, device=device, imgsz=imgsz, persist=True,
                verbose=False, classes=list(KEEP_CLASSES), **track_kw,
            )[0]
            det_ms = (time.perf_counter() - t0) * 1e3

            raw_df = make_df_from_results(results, emitted_idx) if results is not None else None
            df     = _filter_classes(raw_df)   # redundant safety filter

            cam_t_emit = time.perf_counter()
            meas = {                                     # added: per-frame measurements
                "src_frame":   start_frame + src_idx,
                "t_arrive":    t_arrive,
                "t_det_start": t0,
                "t_det_done":  cam_t_emit,               # detection + tracking + table built
                "n_obj":       0 if df is None else len(df),
                "det_pre_ms":  results.speed["preprocess"],
                "det_inf_ms":  results.speed["inference"],
                "det_post_ms": results.speed["postprocess"],
            }
            payload: dict = {
                "perception_df": df.copy() if df is not None else None,
                "frame_idx":     emitted_idx,
                "cam_t_emit":    cam_t_emit,
                "meas":          meas,
            }
            if save_frames:
                payload["frame_bgr"] = frame   # only ~3 MB when explicitly needed

            try:
                cam_queue.put_nowait(payload)
            except Full:                                 # added: drop-oldest kept, but counted
                try:
                    q_drops.append(int(cam_queue.get_nowait()["frame_idx"]))
                except Empty:
                    pass
                try:
                    cam_queue.put_nowait(payload)
                except Full:
                    q_drops.append(emitted_idx)
            rows.append({"frame_idx": emitted_idx, **meas})

            if t_first is None: t_first = cam_t_emit
            t_last      = cam_t_emit
            emitted_idx += 1
            src_idx     += 1
            frame_count += 1
            det_lats.append(det_ms)
            fps_tracker.tick()

            # pacing
            if pace == "original":
                now = time.perf_counter()
                if now < next_due:
                    time.sleep(next_due - now)
                    now = time.perf_counter()
                steps    = max(1, int((now - next_due) / frame_period) + 1)
                next_due += steps * frame_period

        if n_read is not None and not stop_event.is_set():
            assert src_idx >= n_read, \
                f"{video_path}: video ended at frame {start_frame + src_idx}, expected {start_frame + n_read}"

    finally:
        try: cap.release()
        except: pass
        if out_dir is not None:                          # added
            os.makedirs(out_dir, exist_ok=True)
            pd.DataFrame(rows).to_csv(os.path.join(out_dir, "camera_frames.csv"), index=False)
            info.update(emitted=emitted_idx, q_phi_dropped_frames=q_drops, t_end=time.perf_counter())
            with open(os.path.join(out_dir, "camera.json"), "w") as f:
                json.dump(info, f, indent=1, default=float)

    avg_fps = 0.0
    if t_first and t_last and frame_count >= 2:
        dt = t_last - t_first
        avg_fps = (frame_count - 1) / dt if dt > 0 else 0.0

    avg_det = float(np.mean(det_lats)) if det_lats else 0.0
    print(f"[CAM] Done. emitted={emitted_idx} | avg FPS={avg_fps:.2f} | "
          f"det mean={avg_det:.1f}ms")

    if summary_q is not None:
        try:
            summary_q.put_nowait({
                "cam_frames": emitted_idx,
                "cam_fps":    avg_fps,
                "cam_det_mean_ms": avg_det,
            })
        except: pass
