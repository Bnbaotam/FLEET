import sys
from pathlib import Path

import flwr as fl
import numpy as np
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from strategies.server import FEDADAM, FedAdamBN  # noqa: E402

# arrays: [weight (parameter), running_mean (buffer), running_var (buffer)]
INIT = [np.full((2, 3), 0.5, np.float32), np.zeros(3, np.float32), np.ones(3, np.float32)]
SIZES = (30, 10)  # client sample counts -> weights 0.75, 0.25


def _strategies():
    common = dict(initial_parameters=ndarrays_to_parameters(INIT), **FEDADAM, eta=0.1)
    return FedAdamBN(**common, buffer_idx=[1, 2], running_var_idx=[2]), fl.server.strategy.FedAdam(**common)


def _results(arrays_per_client):
    return [(None, FitRes(Status(Code.OK, ""), ndarrays_to_parameters(a), n, {})) for a, n in zip(arrays_per_client, SIZES)]


def _clients(rng, var):
    return [[rng.normal(size=(2, 3)).astype(np.float32), rng.normal(size=3).astype(np.float32), np.full(3, var, np.float32)] for _ in SIZES]


def test_buffers_are_weighted_average_and_parameters_match_stock_fedadam():
    rng = np.random.default_rng(0)
    bn, stock = _strategies()
    for rnd in (1, 2, 3):
        clients = _clients(rng, var=rng.uniform(0.1, 2.0))
        out = parameters_to_ndarrays(bn.aggregate_fit(rnd, _results(clients), [])[0])
        ref = parameters_to_ndarrays(stock.aggregate_fit(rnd, _results(clients), [])[0])
        for i in (1, 2):
            np.testing.assert_allclose(out[i], 0.75 * clients[0][i] + 0.25 * clients[1][i], rtol=1e-6)
            np.testing.assert_array_equal(bn.current_weights[i], out[i])
        np.testing.assert_array_equal(out[0], ref[0])
        assert not np.allclose(out[2], ref[2])  # stock FedAdam steps the buffers instead


def test_running_var_stays_at_client_value_over_rounds():
    rng = np.random.default_rng(1)
    bn, stock = _strategies()
    for rnd in range(1, 21):
        clients = _clients(rng, var=0.01)
        out = parameters_to_ndarrays(bn.aggregate_fit(rnd, _results(clients), [])[0])
        ref = parameters_to_ndarrays(stock.aggregate_fit(rnd, _results(clients), [])[0])
        np.testing.assert_allclose(out[2], 0.01, rtol=1e-6)
    assert (ref[2] < 0).all(), ref[2]  # stock FedAdam overshoots the target (about -0.27 after 20 rounds)
