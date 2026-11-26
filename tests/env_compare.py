"""Compares two processed data files and fails on any difference between them."""

import types

import dill
import numpy as np


def load_env(path):
    with open(path, "rb") as f:  # as src/data/unified_loader.load_env
        return dill.load(f, encoding="latin1")


def compare_envs(a, b) -> dict:
    """Assert that a and b are equal object graphs: same types and attributes everywhere (standardization, attention radius,
    scenes with dt / timesteps / name, nodes with id / type / first_timestep / data arrays, the rotated copies of training
    scenes), arrays exactly equal (NaN = NaN) and functions with the same name, bytecode, constants and names. Returns counts
    and the maximum absolute difference over all float arrays (0 when the assertion passes)."""
    seen = set()
    stats = {"arrays": 0, "array_values": 0, "max_abs_diff": 0.0}

    def rec(a, b, path):
        if (id(a), id(b)) in seen:
            return
        seen.add((id(a), id(b)))
        assert type(a) is type(b), (path, type(a), type(b))
        if isinstance(a, np.ndarray):
            assert a.shape == b.shape and a.dtype == b.dtype, (path, a.shape, b.shape, a.dtype, b.dtype)
            if a.dtype.kind == "f" and a.size:
                assert (np.isnan(a) == np.isnan(b)).all(), path
                diff = np.abs(a - b)[~np.isnan(a)]
                stats["max_abs_diff"] = max(stats["max_abs_diff"], float(diff.max()) if diff.size else 0.0)
            assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), path
            stats["arrays"] += 1
            stats["array_values"] += a.size
        elif isinstance(a, types.FunctionType):
            ca, cb = a.__code__, b.__code__
            assert (a.__name__, ca.co_code, ca.co_consts, ca.co_names) == (b.__name__, cb.co_code, cb.co_consts, cb.co_names), path
        elif isinstance(a, dict):
            assert len(a) == len(b), (path, len(a), len(b))
            for i, ((ka, va), (kb, vb)) in enumerate(zip(a.items(), b.items())):
                rec(ka, kb, f"{path}<key {i}>")
                rec(va, vb, f"{path}[{ka!r}]")
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b), (path, len(a), len(b))
            for i, (x, y) in enumerate(zip(a, b)):
                rec(x, y, f"{path}[{i}]")
        elif not hasattr(a, "__dict__") or isinstance(a, type):
            assert a == b or (a != a and b != b), (path, a, b)
        if hasattr(a, "__dict__") and not isinstance(a, (type, types.FunctionType)):
            assert vars(a).keys() == vars(b).keys(), (path, sorted(vars(a)), sorted(vars(b)))
            for k in vars(a):
                rec(vars(a)[k], vars(b)[k], f"{path}.{k}")

    rec(a, b, "env")
    assert stats["max_abs_diff"] == 0.0, stats
    aug = [s for s in a.scenes if hasattr(s, "augmented")]
    return {"scenes": len(a.scenes), "nodes": sum(len(s.nodes) for s in a.scenes), "augmented_scenes": sum(len(s.augmented) for s in aug),
            "augmented_nodes": sum(len(x.nodes) for s in aug for x in s.augmented), **stats}
