#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

COLUMNS = ["frame", "trackId", "x", "y", "sceneId", "metaId"]
STANDING_FRAMES = 20


def load_and_subsample(path: Path, delta_time: float, orig_fps: float) -> pd.DataFrame:
    """Box centres, every step-th distinct frame, frames re-indexed from 0; rows sorted by (trackId, frame)."""
    df = pd.read_csv(path, header=None, float_precision="round_trip")
    assert df.shape[1] >= 6, f"{path}: a MOT line has at least 6 columns (frame, track_id, x, y, w, h), found {df.shape[1]}"
    df = df.iloc[:, :6].set_axis(["frame_id", "track_id", "x", "y", "w", "h"], axis=1)
    assert all(df[c].dtype.kind == "i" for c in ("frame_id", "track_id")), f"{path}: frame and track ids must be integers"
    assert np.isfinite(df[["x", "y", "w", "h"]].to_numpy(dtype=float)).all(), f"{path}: non-finite box coordinates"
    assert not df.duplicated(["frame_id", "track_id"]).any(), f"{path}: a track has two boxes in one frame"
    df["x"] = df["x"] + df["w"] / 2.0
    df["y"] = df["y"] + df["h"] / 2.0
    step = int(round(delta_time * orig_fps))
    assert step >= 1, (delta_time, orig_fps)
    sampled = sorted(df["frame_id"].unique())[::step]
    df = df[df["frame_id"].isin(sampled)].copy()
    df["frame_id"] = df["frame_id"].map({old: new for new, old in enumerate(sampled)})
    return (df.rename(columns={"frame_id": "frame", "track_id": "trackId"})[["frame", "trackId", "x", "y"]]
            .sort_values(["trackId", "frame"]).reset_index(drop=True))


def identify_tracks(df: pd.DataFrame, min_movement: float) -> tuple[list, list]:
    """(standing, moving) track ids by the displacement between the first and the last position."""
    standing, moving = [], []
    for tid, grp in df.groupby("trackId", sort=False):
        dx = grp["x"].iloc[-1] - grp["x"].iloc[0]
        dy = grp["y"].iloc[-1] - grp["y"].iloc[0]
        (standing if np.hypot(dx, dy) < min_movement else moving).append(tid)
    return standing, moving


def fill_track_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Every track gets every frame between its first and last one; x and y linearly interpolated."""
    out = []
    for tid, grp in df.groupby("trackId", sort=False):
        grp = grp.sort_values("frame").set_index("frame")
        gf = grp.reindex(np.arange(grp.index.min(), grp.index.max() + 1))
        gf[["x", "y"]] = gf[["x", "y"]].interpolate().ffill().bfill()
        gf["trackId"] = tid
        out.append(gf.rename_axis("frame").reset_index())
    return pd.concat(out, ignore_index=True)


def trim_and_filter(df: pd.DataFrame, standing: list, moving: list, min_frames: int) -> pd.DataFrame:
    """Standing tracks cut to their first STANDING_FRAMES frames, then tracks shorter than min_frames dropped."""
    stand = df[df["trackId"].isin(standing)].groupby("trackId", group_keys=False).head(STANDING_FRAMES)
    df = pd.concat([stand, df[df["trackId"].isin(moving)]], ignore_index=True)
    counts = df["trackId"].value_counts()
    return df[df["trackId"].isin(counts[counts >= min_frames].index)].reset_index(drop=True)


def split_objects(df: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train / test by track id."""
    train_ids, test_ids = train_test_split(sorted(df["trackId"].unique()), test_size=test_size, random_state=random_state, shuffle=True)
    assert not set(train_ids) & set(test_ids)
    return df[df["trackId"].isin(train_ids)].reset_index(drop=True), df[df["trackId"].isin(test_ids)].reset_index(drop=True)


def preprocess(path: Path, delta_time: float, orig_fps: float, min_movement: float, min_frames: int, test_size: float,
               random_state: int, scene: str = "scene") -> tuple[pd.DataFrame, pd.DataFrame]:
    assert min_movement >= 0 and min_frames >= 1 and 0 < test_size < 1, (min_movement, min_frames, test_size)
    df = load_and_subsample(path, delta_time, orig_fps)
    standing, moving = identify_tracks(df, min_movement)
    df = trim_and_filter(fill_track_gaps(df), standing, moving, min_frames)
    assert df["trackId"].nunique() >= 2, f"{path}: {df['trackId'].nunique()} track(s) left, a train/test split needs 2"
    assert np.isfinite(df[["x", "y"]].to_numpy()).all()
    # metaId, row order, the smallest integer types, split, sceneId
    df["metaId"] = df["trackId"].map({t: i for i, t in enumerate(sorted(df["trackId"].unique()))})
    df = df.sort_values(["frame", "trackId"]).reset_index(drop=True)
    df[["frame", "trackId"]] = df[["frame", "trackId"]].apply(pd.to_numeric, downcast="integer")
    train, test = split_objects(df, test_size, random_state)
    assert len(train) + len(test) == len(df) and len(train) and len(test)
    return train.assign(sceneId=f"{scene}_train")[COLUMNS], test.assign(sceneId=f"{scene}_test")[COLUMNS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", "-d", type=Path, required=True, help="MOT-format text file of one camera view")
    ap.add_argument("--train-output", type=Path, required=True)
    ap.add_argument("--test-output", type=Path, required=True)
    ap.add_argument("--delta-time", type=float, default=0.1, help="seconds between kept frames")
    ap.add_argument("--orig-fps", type=float, default=10.0, help="frame rate of the tracker output")
    ap.add_argument("--min-movement", type=float, default=20.0, help="pixels: tracks that move less are standing")
    ap.add_argument("--min-frames", type=int, default=20, help="minimum track length in frames")
    ap.add_argument("--test-size", type=float, default=0.2, help="fraction of the tracks in the test file")
    ap.add_argument("--random-state", type=int, default=42, help="seed of the train/test split")
    ap.add_argument("--scene-name", default="scene", help="label stored in the sceneId column as <scene name>_train / _test")
    a = ap.parse_args()
    assert a.train_output.resolve() != a.test_output.resolve(), "--train-output and --test-output are the same file"
    train, test = preprocess(a.data, a.delta_time, a.orig_fps, a.min_movement, a.min_frames, a.test_size, a.random_state, a.scene_name)
    for df, path in ((train, a.train_output), (test, a.test_output)):
        with open(path, "wb") as f:
            pickle.dump(df, f)
        print(f"{path}: {df['trackId'].nunique()} tracks, {len(df)} positions")


if __name__ == "__main__":
    main()
