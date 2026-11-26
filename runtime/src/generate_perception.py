#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd


def generate_packets(
    n_objects: int,
    n_frames:  int,
    fps:       float,
    img_w:     int = 1920,
    img_h:     int = 1080,
    scene_id:  str = "Georgie_NW",
    seed:      int = 42,
):
    """
    Returns a list of packet dicts that look exactly like
    camera_processor.py emits to q_phi.
    """
    rng = np.random.default_rng(seed)

    # Random-walk initial state
    cx = rng.uniform(100, img_w - 100, size=n_objects)
    cy = rng.uniform(100, img_h - 100, size=n_objects)
    vx = rng.uniform(-4, 4, size=n_objects)
    vy = rng.uniform(-4, 4, size=n_objects)

    packets = []
    for f in range(n_frames):
        rows = []
        for i in range(n_objects):
            # Random-walk step with soft wall bounce
            cx[i] += vx[i] + rng.normal(0, 0.3)
            cy[i] += vy[i] + rng.normal(0, 0.3)
            cx[i]  = float(np.clip(cx[i], 50, img_w - 50))
            cy[i]  = float(np.clip(cy[i], 50, img_h - 50))
            if cx[i] <= 50 or cx[i] >= img_w - 50: vx[i] *= -1
            if cy[i] <= 50 or cy[i] >= img_h - 50: vy[i] *= -1

            bw, bh = 60, 120
            x1 = int(cx[i] - bw / 2)
            y1 = int(cy[i] - bh / 2)
            x2 = x1 + bw
            y2 = y1 + bh

            rows.append({
                "frame":     f,
                "trackId":   i,
                "x":         round(cx[i], 3),
                "y":         round(cy[i], 3),
                "bbox":      [x1, y1, x2, y2],      # list (same as make_df_from_results)
                "cls_name":  "PEDESTRIAN",
                "sceneId":   scene_id,
                "metaId":    i,
            })

        perception_df = pd.DataFrame(rows)

        # Dummy black frame (H=108, W=192 — tiny, just a placeholder)
        frame_bgr = np.zeros((108, 192, 3), dtype=np.uint8)

        packets.append({
            "frame_bgr":     frame_bgr,
            "perception_df": perception_df,
            "frame_idx":     f,
            "cam_t_emit":    time.perf_counter(),
        })

    return packets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-objects", type=int,   default=10)
    ap.add_argument("--n-frames",  type=int,   default=300)
    ap.add_argument("--fps",       type=float, default=10.0)
    ap.add_argument("--img-w",     type=int,   default=1920)
    ap.add_argument("--img-h",     type=int,   default=1080)
    ap.add_argument("--scene-id",  default="Georgie_NW")
    ap.add_argument("--seed",      type=int,   default=42)
    ap.add_argument("--out-pkl",   default="data/perception_bench.pkl",
                    help="Pickle of packet list (used by prediction_benchmark.py)")
    ap.add_argument("--out-csv",   default="data/perception_bench.csv",
                    help="Flat CSV of all detections (for inspection)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_pkl) if os.path.dirname(args.out_pkl) else ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv) if os.path.dirname(args.out_csv) else ".", exist_ok=True)

    print(f"Generating {args.n_frames} frames × {args.n_objects} objects "
          f"@ {args.fps} FPS  (scene={args.scene_id}) ...")

    packets = generate_packets(
        n_objects=args.n_objects,
        n_frames=args.n_frames,
        fps=args.fps,
        img_w=args.img_w,
        img_h=args.img_h,
        scene_id=args.scene_id,
        seed=args.seed,
    )

    # Save pickle (used by benchmark)
    with open(args.out_pkl, "wb") as f:
        pickle.dump(packets, f)
    print(f"Saved {len(packets)} packets → {args.out_pkl}")

    # Save flat CSV (for inspection / debugging)
    all_rows = []
    for pkt in packets:
        df = pkt["perception_df"].copy()
        df["frame_idx"] = pkt["frame_idx"]
        # Serialise bbox list to JSON string for CSV storage
        df["bbox"] = df["bbox"].apply(lambda b: json.dumps(b) if isinstance(b, list) else b)
        all_rows.append(df)
    flat = pd.concat(all_rows, ignore_index=True)

    with open(args.out_csv, "w") as f:
        f.write(f"# n_objects={args.n_objects} n_frames={args.n_frames} "
                f"fps={args.fps} scene_id={args.scene_id}\n")
    flat.to_csv(args.out_csv, mode="a", index=False)
    print(f"Saved {len(flat)} rows → {args.out_csv}")
    print(f"Columns: {list(flat.columns)}")


if __name__ == "__main__":
    main()
