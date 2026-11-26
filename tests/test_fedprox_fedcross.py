import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import strategies.client as client_module  # noqa: E402
from strategies.client import FedProxClient  # noqa: E402
from strategies.fedcross import FedCrossStrategy  # noqa: E402
from strategies.server import FEDCROSS  # noqa: E402


# ---- FedProx: g_total = g + mu * (w - w_global) on every trainable parameter ----------------------------------------------
class Tiny(torch.nn.Module):
    """loss = w . x + b, so dloss/dw = x and dloss/db = 1 at every point (the plain gradient is known by hand)."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.b = torch.nn.Parameter(torch.tensor(0.5))
        self.frozen = torch.nn.Parameter(torch.tensor([7.0]), requires_grad=False)

    def loss(self, x):
        return (self.w * x).sum() + self.b


X = torch.tensor([1.0, -1.0])


def _n_hooks(p):
    return len(p._backward_hooks or {})  # None when no hook was ever registered


def _fake_fl_local_train(grads, hook_counts, steps=2, lr=1.0):
    """Stands in for fl_local_train: `steps` SGD steps on Tiny.loss; records the gradients the hooks produce."""
    def train(cfg, model, loader, num_epochs, max_batches):
        for _ in range(steps):
            model.zero_grad()
            if not grads:  # before the first backward: one hook per trainable parameter, none on the frozen one
                hook_counts.update({n: _n_hooks(p) for n, p in model.named_parameters()})
            model.loss(X).backward()
            grads.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p -= lr * p.grad
        return 0.0
    return train


def _run_fedprox(monkeypatch, mu):
    client = object.__new__(FedProxClient)  # skip __init__ (needs data); local_train only uses model, cfg, train_loader, max_batches
    client.model, client.cfg, client.train_loader, client.max_batches = Tiny(), None, None, None
    grads, hook_counts = [], {}
    monkeypatch.setattr(client_module, "fl_local_train", _fake_fl_local_train(grads, hook_counts))
    client.local_train({"proximal_mu": mu})
    return client, grads, hook_counts


def test_fedprox_gradient_is_plain_gradient_plus_mu_times_distance_to_round_start_weights(monkeypatch):
    client, grads, hook_counts = _run_fedprox(monkeypatch, mu=0.5)
    # step 1: w = w_global, so the prox term is 0 -> grad = plain = (x, 1) = ([1, -1], 1); SGD(lr=1) moves w to [0, 3], b to -0.5
    torch.testing.assert_close(grads[0]["w"], torch.tensor([1.0, -1.0]))
    torch.testing.assert_close(grads[0]["b"], torch.tensor(1.0))
    # step 2 (reference must still be the round-start [1, 2], 0.5, not the moved weights):
    # w: [1, -1] + 0.5 * ([0, 3] - [1, 2]) = [0.5, -0.5];  b: 1 + 0.5 * (-0.5 - 0.5) = 0.5
    torch.testing.assert_close(grads[1]["w"], torch.tensor([0.5, -0.5]))
    torch.testing.assert_close(grads[1]["b"], torch.tensor(0.5))
    assert "frozen" not in grads[0] and client.model.frozen.grad is None
    assert hook_counts == {"w": 1, "b": 1, "frozen": 0}, hook_counts  # hook on every trainable parameter only


def test_fedprox_hooks_are_removed_after_local_train(monkeypatch):
    client, _, _ = _run_fedprox(monkeypatch, mu=0.5)
    assert all(_n_hooks(p) == 0 for p in client.model.parameters())
    client.model.zero_grad()
    client.model.loss(X).backward()
    torch.testing.assert_close(client.model.w.grad, X)  # plain gradient again


def test_fedprox_hooks_removed_and_error_propagates_if_training_fails(monkeypatch):
    client = object.__new__(FedProxClient)
    client.model, client.cfg, client.train_loader, client.max_batches = Tiny(), None, None, None

    def boom(*args, **kwargs):
        raise RuntimeError("boom")
    monkeypatch.setattr(client_module, "fl_local_train", boom)
    with pytest.raises(RuntimeError, match="boom"):
        client.local_train({"proximal_mu": 0.5})
    assert all(_n_hooks(p) == 0 for p in client.model.parameters())


@pytest.mark.parametrize("mu", [0.0, -0.1])
def test_fedprox_mu_must_be_positive(monkeypatch, mu):
    with pytest.raises(AssertionError):
        _run_fedprox(monkeypatch, mu=mu)


# ---- FedCross --------------------------------------------------------------------------------------------------------------
# Models are [a0, a1, a2]. similarity = cos(concat(a0, a1)) + cos(a2) (arrays are paired consecutively, an odd last array is its own chunk).
# Chunk 1 is the same direction in a0 and a1, so cos(chunk 1) = cos(a0); scales (3 for model 1) must not matter (cosine, not dot product).
def _model(a0, a2, scale=1.0):
    a0 = scale * np.array(a0, np.float32)
    return [a0, a0.copy(), np.array(a2, np.float32)]


MODELS = [_model([1, 0], [1, 0]), _model([1, 0], [0, 1], scale=3.0), _model([0, 1], [0, 1]), _model([-1, 0], [0, -1])]
SIZES = [10, 20, 30, 40]
# hand-computed (cos chunk 1 + cos chunk 2):   S01 = 1+0, S02 = 0+0, S03 = -1+0, S12 = 0+1, S13 = -1+-1, S23 = 0+-1
SIM = np.array([[0, 1, 0, -1], [1, 0, 1, -2], [0, 1, 0, -1], [-1, -2, -1, 0]], np.float32)
PARTNER = [3, 3, 3, 1]  # least similar other model per row: row0 min(1, 0, -1) -> 3; row1 min(1, 1, -2) -> 3; row2 min(0, 1, -1) -> 3; row3 min(-1, -2, -1) -> 1


def _strategy(alpha):
    return FedCrossStrategy(initial_parameters=ndarrays_to_parameters(MODELS[0]), model_keys=["a0", "a1", "a2"], num_rounds=20,
                            fedcross_alpha=alpha, **FEDCROSS)


def test_fedcross_config_is_lowest_similarity_partner_without_warmup():
    assert FEDCROSS == dict(first_stage_bound=0, collaborative_model_select_strategy=1)


def test_calculate_similarity_matches_hand_values():
    sim = np.array(_strategy(0.99).calculate_similarity(MODELS))
    np.testing.assert_allclose(sim, SIM, atol=1e-6)
    np.testing.assert_array_equal(sim, sim.T)


def test_calculate_similarity_counts_last_chunk_twice_for_an_even_number_of_arrays():
    """Characterisation of the ported behaviour: with 2 arrays (one chunk) the final cosine is added again -> 2 * cos."""
    strategy = FedCrossStrategy(initial_parameters=ndarrays_to_parameters(MODELS[0][:2]), model_keys=["a0", "a1"], num_rounds=20)
    sim = strategy.calculate_similarity([MODELS[0][:2], MODELS[3][:2]])  # chunk 1 of model 0 vs model 3: cos = -1
    assert sim[0][1] == pytest.approx(-2.0, abs=1e-6)


@pytest.mark.parametrize("alpha", [0.99, 0.5])
def test_cross_aggregation_is_alpha_self_plus_one_minus_alpha_least_similar_partner(alpha):
    strategy = _strategy(alpha)
    ids = ["c0", "c1", "c2", "c3"]
    updated = strategy.cross_aggregation(ids, MODELS, strategy.calculate_similarity(MODELS), iteration=1)
    assert list(updated) == ids
    for j, cid in enumerate(ids):
        for layer, new in enumerate(updated[cid]):
            expected = alpha * MODELS[j][layer] + (1 - alpha) * MODELS[PARTNER[j]][layer]
            np.testing.assert_allclose(new, expected, rtol=1e-6, atol=1e-7, err_msg=f"model {j} array {layer}")
            assert new.dtype == np.float32
    if alpha == 0.5:  # literal check: model 0 a0 = 0.5 * [1, 0] + 0.5 * [-1, 0] (model 3); model 3 a2 = 0.5 * [0, -1] + 0.5 * [0, 1] (model 1)
        np.testing.assert_allclose(updated["c0"][0], [0.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(updated["c3"][2], [0.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(updated["c3"][0], [-0.5 + 1.5, 0.0], atol=1e-6)  # 0.5 * [-1, 0] + 0.5 * [3, 0]


def test_cross_aggregation_partner_does_not_depend_on_round():
    strategy = _strategy(0.5)
    sim = strategy.calculate_similarity(MODELS)
    first = strategy.cross_aggregation(["c0", "c1", "c2", "c3"], MODELS, sim, iteration=1)
    for iteration in (2, 3, 7):
        again = strategy.cross_aggregation(["c0", "c1", "c2", "c3"], MODELS, sim, iteration=iteration)
        for cid in first:
            for a, b in zip(first[cid], again[cid]):
                np.testing.assert_array_equal(a, b)


def test_aggregate_fit_global_is_sample_weighted_average_and_locals_are_cross_aggregated():
    alpha = 0.5
    strategy = _strategy(alpha)
    clients = [SimpleNamespace(cid=f"c{j}") for j in range(4)]
    results = [(c, FitRes(Status(Code.OK, ""), ndarrays_to_parameters(m), n, {})) for c, m, n in zip(clients, MODELS, SIZES)]
    params, _ = strategy.aggregate_fit(1, results, [])
    out = parameters_to_ndarrays(params)
    # weights 0.1, 0.2, 0.3, 0.4: a0 = a1 = 0.1 * [1, 0] + 0.2 * [3, 0] + 0.3 * [0, 1] + 0.4 * [-1, 0] = [0.3, 0.3]
    #                             a2 = 0.1 * [1, 0] + 0.2 * [0, 1] + 0.3 * [0, 1] + 0.4 * [0, -1] = [0.1, 0.1]
    for got, want in zip(out, ([0.3, 0.3], [0.3, 0.3], [0.1, 0.1])):
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7)
    for got, want in zip(parameters_to_ndarrays(strategy.global_parameters), out):
        np.testing.assert_array_equal(got, want)  # the stored global model is the returned one
    # first_stage_bound = 0 -> the next-round local models are the cross-aggregated ones (not the global average)
    for j, c in enumerate(clients):
        for layer, local in enumerate(strategy.client_local_models[c.cid]):
            np.testing.assert_allclose(local, alpha * MODELS[j][layer] + (1 - alpha) * MODELS[PARTNER[j]][layer], rtol=1e-6, atol=1e-7)
