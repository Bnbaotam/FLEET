"""Appendix A.5 (camera views outside the federation), known input -> known output: the files of preprocessing/unseen_views.py for
the distributed views, the mean head and the head-only training of run_local.py / strategies/local.py, the --clients 17 argument
checks, tags and hparams, the source-run check, the missing-file stop, round 0 in merge_site_parts and the summarize columns."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "preprocessing"), str(ROOT / "tests")]

import run_fl  # noqa: E402
import run_local  # noqa: E402
import unseen_views  # noqa: E402
from env_compare import compare_envs, load_env  # noqa: E402
from strategies.baselines import PART, merge_site_parts  # noqa: E402
from strategies.local import frozen_init, train_head_only  # noqa: E402
from summarize import ORDER, TAG, clients_file, read_clients, run_key, table  # noqa: E402

DISTRIBUTED = clients_file(ROOT / "configs" / "clients_17_chattanooga.txt")


# ---- the 17 views and their files -----------------------------------------------------------------------------------------
def test_clients_17_is_23_minus_06_in_23_order():
    c17, c23, c06 = read_clients("17"), read_clients("23"), read_clients("06")
    assert c17 == [s for s in c23 if s not in c06] and len(c17) == 17
    assert DISTRIBUTED == [s for s in c17 if s in clients_file(ROOT / "configs" / "clients_23_chattanooga.txt")] == ["Pine_1", "Hwy27_2", "Georgia_1", "Lindsay_1"]


def st06() -> dict:
    return load_env(ROOT / "data/processed_06/Broad_2_test.pkl").standardization


def test_federation_constants():
    st, radius, members = unseen_views.federation_constants(ROOT / "data" / "processed_06")
    assert st == st06() and members == set(read_clients("06")) and list(radius.values()) == [15.0]
    js = json.loads((ROOT / "configs/standardization_06.json").read_text())
    assert st == {"PEDESTRIAN": js["standardization"]}


def test_federation_constants_refuses_differing_files(tmp_path):
    for n in ("A_train.pkl", "A_test.pkl"):
        (tmp_path / n).write_bytes((ROOT / "data/processed_06/Broad_2_test.pkl").read_bytes())
    unseen_views.federation_constants(tmp_path)
    (tmp_path / "B_train.pkl").write_bytes((ROOT / "data/processed_23/Pine_1_test.pkl").read_bytes())  # other constants, no test file
    with pytest.raises(AssertionError, match="train/test pairs"):
        unseen_views.federation_constants(tmp_path)
    (tmp_path / "B_test.pkl").write_bytes((ROOT / "data/processed_23/Pine_1_test.pkl").read_bytes())
    with pytest.raises(AssertionError, match="B_t.*differs"):
        unseen_views.federation_constants(tmp_path)


def test_unseen_env_changes_only_the_standardization_and_refuses_the_federation_constants():
    raw = load_env(ROOT / "data/processed_23/Pine_1_test.pkl")
    env = unseen_views.unseen_env(raw, st06(), raw.attention_radius)
    assert env.standardization == st06() != raw.standardization  # raw untouched
    with pytest.raises(AssertionError, match="already carries"):
        unseen_views.unseen_env(env, st06(), raw.attention_radius)
    with pytest.raises(AssertionError):
        unseen_views.unseen_env(raw, st06(), {k: 2 * r for k, r in raw.attention_radius.items()})


@pytest.fixture(scope="module")
def unseen_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("unseen")
    subprocess.run([sys.executable, str(ROOT / "preprocessing/unseen_views.py"), "--clients", str(ROOT / "configs/clients_17_chattanooga.txt"),
                    "--out_dir", str(out)], check=True, capture_output=True)
    return out


def test_files_of_the_distributed_views(unseen_dir):
    """Every file = the processed file of the 23-view setting with the standardization of data/processed_06, equal in everything else."""
    assert sorted(p.name for p in unseen_dir.iterdir()) == sorted(f"{v}_{s}.pkl" for v in DISTRIBUTED for s in ("train", "test"))
    for view in DISTRIBUTED:
        for split in ("train", "test"):
            env, raw = load_env(unseen_dir / f"{view}_{split}.pkl"), load_env(ROOT / f"data/processed_23/{view}_{split}.pkl")
            assert env.standardization == st06() != raw.standardization, (view, split)
            env.standardization = raw.standardization
            stats = compare_envs(env, raw)
            assert stats["nodes"] > 0 and stats["augmented_scenes"] == (24 * stats["scenes"] if split == "train" else 0), (view, split, stats)


def test_unseen_views_refuses_missing_files_and_member_views(tmp_path):
    def run(*views):
        (tmp_path / "c.txt").write_text("\n".join(views) + "\n")
        return subprocess.run([sys.executable, str(ROOT / "preprocessing/unseen_views.py"), "--clients", str(tmp_path / "c.txt"),
                               "--out_dir", str(tmp_path / "o")], capture_output=True, text=True)
    r = run(*read_clients("17"))  # only the 4 distributed views are present
    assert r.returncode != 0 and "26 of the 34 processed files are missing" in r.stderr and "C042_test.pkl" in r.stderr
    r = run("Pine_1", "Broad_2")
    assert r.returncode != 0 and "views of the federation" in r.stderr and "Broad_2" in r.stderr
    assert not (tmp_path / "o").exists()


# ---- mean head ------------------------------------------------------------------------------------------------------------
class Toy(nn.Module):
    def __init__(self, momentum=0.1):
        super().__init__()
        self.encoder = nn.Linear(3, 2)
        self.flow = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2, momentum=momentum))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.1)


def toy_states(n=3):
    torch.manual_seed(0)
    enc = Toy().encoder.state_dict()
    states = []
    for i in range(n):
        m = Toy()
        m.encoder.load_state_dict(enc)
        m.flow[1].running_mean.fill_(float(i))
        m.flow[1].num_batches_tracked.fill_(10 * i + 1)
        states.append(m.state_dict())
    return states


def test_mean_head_on_toy_state_dicts():
    states, model = toy_states(), Toy()
    shared = run_local.exchanged_keys("fedrep", model)
    assert shared == ["encoder.bias", "encoder.weight"]
    out = run_local.mean_head(states, shared, model)
    assert list(out) == list(model.state_dict())
    for k in shared:
        assert torch.equal(out[k], states[0][k])
    torch.testing.assert_close(out["flow.0.weight"], (states[0]["flow.0.weight"] + states[1]["flow.0.weight"] + states[2]["flow.0.weight"]) / 3)
    assert torch.equal(out["flow.1.running_mean"], torch.full((2,), 1.0))
    assert out["flow.1.num_batches_tracked"].dtype == torch.long and out["flow.1.num_batches_tracked"].item() == 21  # max
    model.load_state_dict(out)


def test_mean_head_refuses_differing_shared_keys_and_bn_without_momentum():
    states, model = toy_states(), Toy()
    states[1]["encoder.weight"] = states[1]["encoder.weight"] + 1e-7
    with pytest.raises(AssertionError, match="shared key encoder.weight differs"):
        run_local.mean_head(states, run_local.exchanged_keys("fedrep", model), model)
    with pytest.raises(AssertionError, match="momentum"):
        run_local.mean_head(toy_states(), ["encoder.weight"], Toy(momentum=None))
    with pytest.raises(AssertionError, match="keys"):
        run_local.mean_head([*toy_states(2), {}], ["encoder.weight"], Toy())


# ---- head-only training -----------------------------------------------------------------------------------------------------
def toy_client(sneak=False):
    """local_train as fl_local_train: the model's own Adam (zero_grad, backward, step) and a fresh Adam step on all parameters."""
    model = Toy()
    model.load_state_dict(toy_states()[0])

    def local_train(config):
        fresh = torch.optim.Adam(model.parameters(), lr=0.1)
        fresh.zero_grad()
        model.optimizer.zero_grad()
        loss = model.flow(model.encoder(torch.randn(8, 3))).pow(2).sum()
        loss.backward()
        model.optimizer.step()
        fresh.step()
        if sneak:
            with torch.no_grad():
                model.encoder.bias.add_(1.0)
        return loss.item()
    return SimpleNamespace(model=model, local_train=local_train)


def test_head_only_freezes_exactly_the_shared_keys():
    client = toy_client()
    before = {k: v.clone() for k, v in client.model.state_dict().items()}
    frozen = frozen_init(client.model, "fedper")
    assert list(frozen) == ["encoder.bias", "encoder.weight"]
    for _ in range(2):
        train_head_only(client, frozen)
    after = client.model.state_dict()
    assert all(torch.equal(after[k], before[k]) for k in frozen)
    assert all(not torch.equal(after[k], before[k]) for k in ("flow.0.weight", "flow.0.bias", "flow.1.weight", "flow.1.bias", "flow.1.running_mean"))
    assert all(p.requires_grad and p.grad is None for p in client.model.parameters())


def test_head_only_detects_a_changed_shared_tensor_and_refuses_shared_buffers():
    client = toy_client(sneak=True)
    with pytest.raises(AssertionError, match="frozen shared tensor changed"):
        train_head_only(client, frozen_init(client.model, "fedrep"))
    model = Toy()
    model.encoder.register_buffer("stat", torch.zeros(1))
    with pytest.raises(AssertionError):
        frozen_init(model, "fedrep")


# ---- argument checks, tags, hparams -----------------------------------------------------------------------------------------
def ns(**kw):
    return argparse.Namespace(**{"clients": "17", "rounds": 0, "head": None, "head_only": False, "init_from": "x", "max_batches": None,
                                 "per_camera": False, "data_dir": None, **kw})


@pytest.mark.parametrize("src,kw,strategy,tag", [
    ("fedavg", {}, "fedavgzs", "fedavgzs-17c"),
    ("fedrep", {"head": "Broad_2"}, "fedrepzs", "fedrepzs-17c-dBroad_2"),
    ("fedper", {"head": "Pine_2", "max_batches": 2}, "fedperzs", "smoke-fedperzs-17c-dPine_2"),
    ("fedavg", {"rounds": 5}, "fedavgft", "fedavgft-17c"),
    ("fedrep", {"rounds": 5, "head": "mean", "head_only": True}, "fedrepad", "fedrepad-17c"),
    ("fedper", {"rounds": 1, "head": "mean", "head_only": True}, "fedperad", "fedperad-17c")])
def test_unseen_strategy_and_tag(src, kw, strategy, tag):
    a = ns(**kw)
    assert run_local.unseen_strategy(a, src) == strategy and run_local.run_tag(a, strategy) == tag


@pytest.mark.parametrize("src,kw,match", [
    ("fedavg", {"head": "Broad_2"}, "--head"), ("fedavg", {"rounds": 5, "head": "mean", "head_only": True}, "--head"),
    ("fedrep", {}, "--head"), ("fedper", {"rounds": 5, "head_only": True}, "--head"),
    ("fedrep", {"rounds": 5, "head": "mean"}, "--head_only"), ("fedrep", {"head": "Broad_2", "head_only": True}, "--head_only"),
    ("fedavg", {"rounds": 5, "head_only": True}, "--head_only"),
    ("fedrep", {"head": "mean"}, "donor intersection"), ("fedrep", {"head": "Pine_1"}, "donor intersection"),
    ("fedrep", {"rounds": 5, "head": "Broad_2", "head_only": True}, "mean"), ("fedavg", {"rounds": -1}, "-1")])
def test_unseen_strategy_rejects(src, kw, match):
    with pytest.raises(AssertionError, match=match):
        run_local.unseen_strategy(ns(**kw), src)


def test_hparams_and_data_dir(tmp_path):
    a = ns(rounds=5, head="mean", head_only=True, init_from="r")
    assert run_local.run_hparams(a) == {"init_from": "r", "mode": "adapt", "head": "mean", "head_only": True, "data": "processed_17_unseen"}
    assert run_local.run_hparams(ns(init_from="r")) == {"init_from": "r", "mode": "zeroshot", "head": None, "head_only": False,
                                                        "data": "processed_17_unseen"}
    assert run_local.run_hparams(ns(init_from="r", data_dir=tmp_path / "mine"))["data"] == "mine"
    assert run_local.run_hparams(ns(clients="23", rounds=5, init_from="r")) == {"init_from": "r"}
    assert run_local.run_hparams(ns(clients="06", rounds=20, init_from=None)) is None
    assert run_local.local_data_dir(ns()) == ROOT / "data" / "processed_17_unseen"
    assert run_local.local_data_dir(ns(clients="06", init_from=None, per_camera=True)) == ROOT / "data" / "processed_per_camera"
    assert run_local.local_data_dir(ns(clients="23", init_from=None, per_camera=True, data_dir=tmp_path)) == tmp_path
    assert run_local.local_data_dir(ns(clients="23", init_from=None)) == ROOT / "data" / "processed_23"


@pytest.mark.parametrize("extra", [["--rounds", "0"], ["--head", "Broad_2"], ["--head_only"]])
def test_17_only_flags_refused_with_06(monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["run_local.py", "--clients", "06", "--seed", "0", *extra])
    with pytest.raises(AssertionError, match="--clients 17 only"):
        run_local.main()


@pytest.mark.parametrize("extra, match", [([], "--init_from is required"), (["--per_camera", "--init_from", "x"], "--per_camera")])
def test_17_needs_a_source_and_no_per_camera(monkeypatch, extra, match):
    monkeypatch.setattr(sys, "argv", ["run_local.py", "--clients", "17", "--seed", "0", "--rounds", "0", *extra])
    with pytest.raises(AssertionError, match=match):
        run_local.main()


def test_17_stops_on_missing_files(tmp_path, monkeypatch):
    """Default folder data/processed_17_unseen (here under a monkeypatched root, empty): every missing file is listed with the hint,
    before the source run is read or any process starts."""
    monkeypatch.setattr(run_fl, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_local.py", "--clients", "17", "--seed", "0", "--rounds", "0", "--init_from", "x"])
    with pytest.raises(FileNotFoundError, match="34 of the 34 processed files are missing") as e:
        run_local.main()
    assert str(tmp_path / "data" / "processed_17_unseen") in str(e.value) and str(e.value).endswith("preprocessing/unseen_views.py or pass --data_dir.")
    assert not (tmp_path / "runs").exists()


def test_source_must_be_a_finished_six_intersection_run_of_the_seed(tmp_path, monkeypatch):
    runs = {"avg": ("fedavg-06c", 0), "rep": ("fedrep-06c", 0), "per1": ("fedper-06c", 1), "loc": ("local-06c", 0),
            "ft": ("fedavgft-06c", 0), "prox": ("fedprox-06c", 0), "smk": ("smoke-fedrep-06c", 0), "a23": ("fedavg-23c", 0),
            "mix": ("fedavg-06c", 0)}
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "index.jsonl").write_text("".join(json.dumps({"run_id": r, "tag": t, "seed": s}) + "\n" for r, (t, s) in runs.items()))
    for r in runs:
        (tmp_path / "runs" / r).mkdir()
        sites = [*read_clients("06")[:-1], "Pine_1"] if r == "mix" else read_clients("06")
        pd.DataFrame({"intersection": sites}).to_csv(tmp_path / "runs" / r / "metrics.csv", index=False)
    monkeypatch.setattr(run_local, "ROOT", tmp_path)
    assert run_local.source_strategy("rep", 0, smoke=False) == "fedrep" and run_local.source_strategy("per1", 1, smoke=False) == "fedper"
    assert run_local.source_strategy("avg", 0, smoke=True) == "fedavg"
    with pytest.raises(AssertionError, match="differs from the seed 0"):
        run_local.source_strategy("avg", 1, smoke=False)
    for r in ("loc", "ft", "prox", "a23"):
        with pytest.raises(AssertionError, match="not a finished six-intersection run"):
            run_local.source_strategy(r, 0, smoke=True)
    with pytest.raises(AssertionError, match="smoke source needs --max_batches"):
        run_local.source_strategy("smk", 0, smoke=False)
    assert run_local.source_strategy("smk", 0, smoke=True) == "fedrep"
    with pytest.raises(AssertionError):  # a source run on other sites
        run_local.source_strategy("mix", 0, smoke=False)


# ---- round 0 -------------------------------------------------------------------------------------------------------------------
def write_parts(d, sites, rounds):
    for s in sites:
        (d / s).mkdir()
        (d / s / PART).write_text("".join(json.dumps({"round": r, "wall_s": 2.0, "metrics": {
            "intersection": s, "n_train_windows": 10, "ade": 1.0, "fde": 2.0, "ade_legacy": 1.0, "fde_legacy": 2.0, "inference_time": 0.1,
            "n_test_windows": 5, "eval_time_s": 1.0}}) + "\n" for r in rounds))


def test_merge_site_parts_accepts_round_0_only_for_zero_rounds(tmp_path):
    write_parts(tmp_path, ["a", "b"], [0])
    merge_site_parts(tmp_path, "rid", "fedavgzs", 1, 0, ["a", "b"])
    m = pd.read_csv(tmp_path / "metrics.csv")
    assert set(m["round"]) == {0} and len(m) == 2 and m.phase.eq("evaluate").all() and not (tmp_path / "a" / PART).exists()
    assert pd.read_csv(tmp_path / "rounds.csv")["round"].tolist() == [0]
    (tmp_path / "x").mkdir()
    write_parts(tmp_path / "x", ["a"], [0])
    with pytest.raises(AssertionError):
        merge_site_parts(tmp_path / "x", "rid", "fedavgft", 1, 1, ["a"])


# ---- TAG and summarize ----------------------------------------------------------------------------------------------------------
def test_new_tags():
    keys = {t: run_key(TAG.fullmatch(t)) for t in ("fedrepzs-17c-dBroad_2", "smoke-fedperzs-17c-dPine_2", "fedavgzs-17c", "fedavgft-17c",
                                                   "fedrepad-17c", "fedperad-17c", "fedrepzs-17c-ds110_camera_basler_south1_8mm")}
    assert list(keys.values()) == ["fedrepzs-dBroad_2", "fedperzs-dPine_2", "fedavgzs", "fedavgft", "fedrepad", "fedperad",
                                   "fedrepzs-ds110_camera_basler_south1_8mm"]
    assert TAG.fullmatch("fedrepzs-17c-dBroad_2")["nn"] == "17"
    for bad in ("fedrepzs-17c", "fedavg-06c-dBroad_2", "fedavgzs-17c-dBroad_2", "fedrepzs-06c-dBroad_2"):
        with pytest.raises(AssertionError):
            run_key(TAG.fullmatch(bad))


def test_release_tags_parse_as_before():
    for tag, key in (("fedavg-06c", "fedavg"), ("smoke-fedprox-23c-mu0.001", "fedprox"), ("local-06c-r25-pc", "local-r25-pc"),
                     ("fedavgft-06c", "fedavgft"), ("central-23c", "central")):
        m = TAG.fullmatch(tag)
        assert run_key(m) == key and m["scaled"] is None and m["donor"] is None, tag


def test_summarize_17_zero_shot_donor_and_adaptation_columns():
    """Zero-shot runs (round 0) and adaptation runs (rounds 1..R) in one table, one column per donor head, in ORDER."""
    runs = (("z", "fedavgzs-17c", [0], (4, 6)), ("d1", "fedrepzs-17c-dBroad_2", [0], (5, 7)), ("d2", "fedrepzs-17c-dPine_2", [0], (1, 3)),
            ("f", "fedavgft-17c", [1, 2], (3, 5)), ("ad", "fedrepad-17c", [1, 2], (2, 2)))
    df = pd.DataFrame([{"run_id": r, "tag": tag, "strategy": tag.split("-")[0], "phase": "evaluate", "split": "test", "intersection": c,
                        "round": rnd, "ade_px": v + rnd, "fde_px": 2 * v, "ade_legacy_px": v, "fde_legacy_px": v}
                       for r, tag, rounds, vals in runs for rnd in rounds for c, v in zip("AB", vals)])
    out = table(df, "17", ["A", "B"], "evaluate", "test", "final", False, list(ORDER)).splitlines()
    assert out[out.index("ADE\tfedavgzs\tfedavgft\tfedrepzs-dBroad_2\tfedrepzs-dPine_2\tfedrepad") + 3] == \
        "Average\t5.000 (n=1)\t6.000 (n=1)\t6.000 (n=1)\t2.000 (n=1)\t4.000 (n=1)"  # final round: 0 for zero-shot, 2 for adaptation
    assert "# fedrepzs-dPine_2 [fedrepzs-17c-dPine_2]: n=1 runs: d2" in out
    with pytest.raises(AssertionError, match="runs of one strategy differ in their number of rounds"):
        table(pd.concat([df, df[df.run_id == "f"].assign(run_id="f3", round=lambda x: x["round"] + 1)]), "17", ["A", "B"], "evaluate", "test",
              "final", False, list(ORDER))
