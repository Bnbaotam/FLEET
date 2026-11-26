import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "preprocessing"), str(ROOT / "tests")]
import compute_standardization as cs  # noqa: E402
import preprocess_tracks as pt  # noqa: E402
import process_data as pdata  # noqa: E402
from env_compare import compare_envs, load_env  # noqa: E402

RAW = ROOT / "data" / "raw_trajectories"
CLIENTS = {nn: pdata.clients_file(ROOT / f"configs/clients_{nn}.txt") for nn in ("06", "23", "23_chattanooga")}


# ---- step 1: preprocess_tracks ------------------------------------------------------------------------------------------
def _mot(tmp_path) -> Path:
    """Synthetic MOT file (frames 5..34, boxes 10 x 20 px, 4 extra columns). Track 1 moves 5 px per frame over 25 frames with
    frames 10 and 11 missing; track 2 stands (moves 0.1 px per frame) for 30 frames; track 3 moves but has 10 frames only;
    track 4 moves 3 px per frame over 22 frames."""
    rows = []
    for k in range(25):
        if 5 + k not in (10, 11):
            rows.append((5 + k, 1, 100 + 5 * k, 50))
    rows += [(5 + k, 2, 300 + 0.1 * k, 80) for k in range(30)]
    rows += [(5 + k, 3, 500 + 8 * k, 90) for k in range(10)]
    rows += [(8 + k, 4, 700, 60 + 3 * k) for k in range(22)]
    path = tmp_path / "tracks.txt"
    path.write_text("".join(f"{f},{t},{x},{y},10,20,1,1,1,0.9\n" for f, t, x, y in sorted(rows)))
    return path


def test_preprocess_tracks_on_a_synthetic_mot_file(tmp_path):
    train, test = pt.preprocess(_mot(tmp_path), 0.1, 10.0, 20.0, 20, 0.2, 42)
    df = pd.concat([train, test])
    assert list(df.columns) == pt.COLUMNS and set(train.sceneId) == {"scene_train"} and set(test.sceneId) == {"scene_test"}
    for part in (train, test):  # rows sorted by (frame, trackId)
        assert part.equals(part.sort_values(["frame", "trackId"]).reset_index(drop=True))
    assert sorted(df.trackId.unique()) == [1, 2, 4]  # track 3 (10 frames) is dropped
    lens = df.groupby("trackId").size().to_dict()
    assert lens == {1: 25, 2: 20, 4: 22}  # the standing track 2 is cut to its first 20 frames; track 1 is gap-filled
    t1, t2, t4 = (df[df.trackId == t].sort_values("frame") for t in (1, 2, 4))
    assert list(t1.frame) == list(range(25)) and list(t2.frame) == list(range(20)) and list(t4.frame) == list(range(3, 25))  # frames 5.. -> 0..
    np.testing.assert_allclose(t1.x, 105 + 5 * np.arange(25))  # box centre x + w / 2, frames 10 and 11 interpolated linearly
    np.testing.assert_allclose(t1.y, 60.0)  # y + h / 2
    np.testing.assert_allclose(t2.x, 305 + 0.1 * np.arange(20))
    np.testing.assert_allclose(t4.y, 70 + 3 * np.arange(22))
    assert dict(zip(df.trackId, df.metaId)) == {1: 0, 2: 1, 4: 2}  # rank among the kept track ids
    # split by track: disjoint, complete, deterministic, = train_test_split of the sorted track ids
    assert not set(train.trackId) & set(test.trackId) and len(train) + len(test) == len(df)
    want_train, want_test = train_test_split([1, 2, 4], test_size=0.2, random_state=42, shuffle=True)
    assert set(train.trackId) == set(want_train) and set(test.trackId) == set(want_test)
    again = pt.preprocess(_mot(tmp_path), 0.1, 10.0, 20.0, 20, 0.2, 42)
    assert train.equals(again[0]) and test.equals(again[1])


def test_preprocess_tracks_subsamples_frames(tmp_path):
    train, test = pt.preprocess(_mot(tmp_path), 0.2, 10.0, 20.0, 5, 0.2, 42)  # every 2nd distinct frame: 5, 7, ..., 33 -> 0..14
    df = pd.concat([train, test])
    assert df.frame.min() == 0 and df.frame.max() == 14 and sorted(df.trackId.unique()) == [1, 2, 3, 4]
    np.testing.assert_allclose(df[df.trackId == 1].sort_values("frame").x, 105 + 10 * np.arange(13))


def test_preprocess_tracks_cli_writes_the_two_pickles(tmp_path):
    out = [tmp_path / "v_train.pkl", tmp_path / "v_test.pkl"]
    subprocess.run([sys.executable, str(ROOT / "preprocessing/preprocess_tracks.py"), "--data", str(_mot(tmp_path)),
                    "--train-output", str(out[0]), "--test-output", str(out[1])], check=True, capture_output=True)
    want = pt.preprocess(_mot(tmp_path), 0.1, 10.0, 20.0, 20, 0.2, 42)
    for path, df in zip(out, want):
        with open(path, "rb") as f:
            assert pickle.load(f).equals(df)


def test_preprocess_tracks_rejects_bad_input(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("1,1,0,0,10,20\n1,1,5,5,10,20\n2,1,0,0,10,20\n")  # two boxes of track 1 in frame 1
    with pytest.raises(AssertionError, match="two boxes"):
        pt.preprocess(path, 0.1, 10.0, 20.0, 1, 0.2, 42)
    path.write_text("1,1,0,0,10\n")
    with pytest.raises(AssertionError, match="6 columns"):
        pt.preprocess(path, 0.1, 10.0, 20.0, 1, 0.2, 42)


def test_split_of_the_distributed_raw_trajectories():
    """For every distributed view, train_test_split(sorted(all track ids), 0.2, 42) gives the stored train / test membership."""
    for cam in CLIENTS["23_chattanooga"]:
        train, test = (pdata.read_raw(RAW, cam, s) for s in ("train", "test"))
        want_train, want_test = train_test_split(sorted(set(train.trackId) | set(test.trackId)), test_size=0.2, random_state=42, shuffle=True)
        assert set(train.trackId) == set(want_train) and set(test.trackId) == set(want_test), cam


# ---- step 2: compute_standardization ------------------------------------------------------------------------------------
def _raw(tracks: dict) -> pd.DataFrame:
    """A raw trajectory frame: {trackId: (first frame, [(x px, y px), ...])}, one scene."""
    rows = [{"frame": f0 + i, "trackId": tid, "x": x, "y": y, "sceneId": "s_train", "metaId": tid}
            for tid, (f0, xy) in tracks.items() for i, (x, y) in enumerate(xy)]
    return pd.DataFrame(rows).sort_values(["frame", "trackId"], kind="stable").reset_index(drop=True)


def test_constants_recipe_on_a_known_input():
    # track 1: x = 0, 50, 150 px (0, 1, 3 units), y constant; track 2: x = 100, 50 px (2, 1 units); track 3 has a gap and is dropped
    df = _raw({1: (0, [(0, 10), (50, 10), (150, 10)]), 2: (1, [(100, 20), (50, 20)]), 3: (0, [(0, 0), (5, 5)])})
    df = df[~((df.trackId == 3) & (df.frame == 1))]
    df = pd.concat([df, pd.DataFrame([{"frame": 2, "trackId": 3, "x": 5.0, "y": 5.0, "sceneId": "s_train", "metaId": 3}])], ignore_index=True)
    env = pdata.process_file(df, None, is_train=False)
    assert sorted(n.id for n in env.scenes[0].nodes) == ["1", "2"]  # 3 has non-consecutive frames
    c = cs.pooled_constants([env]).set_index(["entity", "dim"])
    x = np.array([0, 1, 3, 2, 1.0])
    vx = np.r_[np.gradient([0, 1, 3.0], 0.1), np.gradient([2, 1.0], 0.1)]  # central differences: 10, 15, 20 and -10, -10
    ax = np.r_[np.gradient(np.gradient([0, 1, 3.0], 0.1), 0.1), np.gradient(np.gradient([2, 1.0], 0.1), 0.1)]
    np.testing.assert_allclose(vx, [10, 15, 20, -10, -10])
    assert c.loc[("position", "x"), "mean"] == round(x.mean(), 6) and c.loc[("position", "x"), "std"] == round(x.std(), 6)
    assert c.loc[("velocity", "x"), "mean"] == 5.0 and c.loc[("velocity", "x"), "std"] == round(vx.std(), 6)
    assert c.loc[("velocity", "y"), "mean"] == 0.0 and c.loc[("velocity", "y"), "std"] == 0.0
    assert c.loc[("acceleration", "x"), "std"] == round(ax.std(), 6) and c.loc[("position", "y"), "mean"] == round(np.mean([.2] * 3 + [.4] * 2), 6)
    assert not c.mean_set_to_0.any() and (c.n_rows == 5).all()
    flipped = cs.pooled_constants([pdata.process_file(df.assign(x=-df.x), None, is_train=False)]).set_index(["entity", "dim"])
    assert flipped.loc[("velocity", "x"), "mean_set_to_0"] and flipped.loc[("velocity", "x"), "mean"] == 0.0  # negative mean -> 0
    assert flipped.loc[("velocity", "x"), "mean_raw"] == pytest.approx(-5.0)
    std = cs.as_standardization(c.reset_index())
    assert std["PEDESTRIAN"]["velocity"]["x"] == {"mean": 5.0, "std": round(vx.std(), 6)}


def _stored(nn: str, cam: str) -> dict:
    return load_env(ROOT / f"data/processed_{nn}/{cam}_test.pkl").standardization


def test_constants_of_the_six_intersections():
    """Recomputed from the raw trajectories == configs/standardization_06.json == stored in every file of data/processed_06."""
    js = json.loads((ROOT / "configs/standardization_06.json").read_text())
    assert js == {"cameras": CLIENTS["06"], **cs.constants(RAW, CLIENTS["06"])}
    for cam in CLIENTS["06"]:
        for split in ("train", "test"):
            assert load_env(ROOT / f"data/processed_06/{cam}_{split}.pkl").standardization == {"PEDESTRIAN": js["standardization"]}, (cam, split)


def test_constants_of_the_23_views_are_stored_in_the_distributed_files():
    """configs/standardization_23.json (computed from the training files of all 23 views) == stored in every file of data/processed_23."""
    js = json.loads((ROOT / "configs/standardization_23.json").read_text())
    assert js["cameras"] == CLIENTS["23"] and js["n_rows"] > 0
    for cam in CLIENTS["23_chattanooga"]:
        for split in ("train", "test"):
            assert load_env(ROOT / f"data/processed_23/{cam}_{split}.pkl").standardization == {"PEDESTRIAN": js["standardization"]}, (cam, split)


def test_standardization_json_checks():
    std = json.loads((ROOT / "configs/standardization_06.json").read_text())["standardization"]
    assert pdata.camera_standardization({"per_camera": {"A": {"n_rows": 3, "standardization": std}}}, ["A"]) == {"A": {"PEDESTRIAN": std}}
    with pytest.raises(AssertionError, match="lack"):
        pdata.camera_standardization({"per_camera": {"A": {"n_rows": 3, "standardization": std}}}, ["A", "B"])
    bad = json.loads(json.dumps(std))
    bad["velocity"]["x"]["std"] = 0.0
    with pytest.raises(AssertionError):
        pdata.check_standardization(bad)


# ---- step 3: process_data ------------------------------------------------------------------------------------------------
def test_process_data_rebuilds_the_distributed_files_of_one_camera(tmp_path):
    clients = tmp_path / "clients.txt"
    clients.write_text("# one camera\nLindsay_2\n")
    subprocess.run([sys.executable, str(ROOT / "preprocessing/process_data.py"), "--raw_dir", str(RAW), "--clients", str(clients),
                    "--standardization", str(ROOT / "configs/standardization_06.json"), "--out_dir", str(tmp_path / "out")],
                   check=True, capture_output=True)
    for split, aug in (("train", 24), ("test", 0)):
        stats = compare_envs(load_env(tmp_path / "out" / f"Lindsay_2_{split}.pkl"), load_env(ROOT / f"data/processed_06/Lindsay_2_{split}.pkl"))
        assert stats["max_abs_diff"] == 0 and stats["nodes"] > 0 and stats["augmented_scenes"] == aug * stats["scenes"], stats
        assert stats["augmented_nodes"] == aug * stats["nodes"], stats


def test_compare_envs_catches_any_difference():
    df = _raw({1: (0, [(0, 10), (50, 10), (150, 10)]), 2: (1, [(100, 20), (50, 20)])})
    a, b = pdata.process_file(df.copy(), {"x": 1}, is_train=True), pdata.process_file(df.copy(), {"x": 1}, is_train=True)
    assert compare_envs(a, b)["augmented_scenes"] == 24
    with pytest.raises(AssertionError):
        compare_envs(a, pdata.process_file(df.copy(), {"x": 2}, is_train=True))  # standardization
    b.scenes[0].augmented[5].nodes[1].data.data[0, 3] += 1e-12  # one value of one rotated copy
    with pytest.raises(AssertionError):
        compare_envs(a, b)
    c = pdata.process_file(df.copy(), {"x": 1}, is_train=True)
    c.scenes[0].nodes[0].first_timestep = 1
    with pytest.raises(AssertionError):
        compare_envs(a, c)


def test_process_data_must_run_as_a_script():
    assert pdata.augment.__module__ == "process_data"  # imported: main() would refuse to pickle aug_func
