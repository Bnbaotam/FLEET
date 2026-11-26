"""Flower server strategy that implements FedCross for FlowChain."""

from collections import OrderedDict
from typing import Dict, List, Tuple, Optional

import flwr as fl
import numpy as np
import torch
import torch.nn.functional as F
from flwr.common import FitIns, Parameters, ndarrays_to_parameters, parameters_to_ndarrays


def ndarrays_to_torch_state_dict(
    keys: List[str], ndarrays: List[np.ndarray]
) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict(
        (k, torch.from_numpy(np.asarray(v))) for k, v in zip(keys, ndarrays)
    )


def torch_state_dict_to_ndarrays(state_dict: "OrderedDict[str, torch.Tensor]") -> List[np.ndarray]:
    return [v.detach().cpu().numpy() for v in state_dict.values()]


def weighted_average_ndarrays(
    results: List[Tuple[List[np.ndarray], int]]
) -> List[np.ndarray]:
    total_examples = sum(num_examples for _, num_examples in results)
    if total_examples == 0:
        raise ValueError("No examples received for aggregation.")

    aggregated = []
    num_layers = len(results[0][0])

    for layer_idx in range(num_layers):
        weighted_sum = None
        for params, num_examples in results:
            layer = np.asarray(params[layer_idx])
            if weighted_sum is None:
                weighted_sum = layer * num_examples
            else:
                weighted_sum += layer * num_examples
        aggregated.append(weighted_sum / total_examples)

    return aggregated


class FedCrossStrategy(fl.server.strategy.Strategy):
    """
    Faithful FlowChain adaptation of FedCross.

    Preserved original FedCross behavior:
    1. Maintain one local model per client.
    2. Each selected client trains starting from its own local model.
    3. Aggregate uploaded models into a global model.
    4. Compute pairwise cosine similarity between client models.
    5. After warmup, cross-aggregate each client's model with one collaborator model.

    Notes:
    - Best used with full participation (all clients every round).
    - Evaluation is performed on the global model, matching the original servercross.py behavior.
    """

    def __init__(
        self,
        initial_parameters: Parameters,
        model_keys: List[str],
        num_rounds: int,
        min_available_clients: int = 8,
        min_fit_clients: int = 8,
        min_evaluate_clients: int = 8,
        first_stage_bound: int = 0,
        fedcross_alpha: float = 0.99,
        collaborative_model_select_strategy: int = 1,
    ) -> None:
        super().__init__()
        self.initial_parameters = initial_parameters
        self.global_parameters = initial_parameters
        self.model_keys = model_keys
        self.num_rounds = num_rounds

        self.min_available_clients = min_available_clients
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients

        self.first_stage_bound = first_stage_bound
        self.cross_alpha = fedcross_alpha
        self.collaborative_model_select_strategy = collaborative_model_select_strategy

        # FedCross state
        self.client_local_models: Dict[str, List[np.ndarray]] = {}
        self.last_fit_client_order: List[str] = []

    def __repr__(self) -> str:
        return "FedCrossStrategy"

    def initialize_parameters(
        self, client_manager: fl.server.client_manager.ClientManager
    ) -> Optional[Parameters]:
        return self.initial_parameters

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        return self.min_fit_clients, self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        return self.min_evaluate_clients, self.min_available_clients

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ):
        clients = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_available_clients,
        )

        fit_configurations = []

        global_ndarrays = parameters_to_ndarrays(self.global_parameters)

        for client in clients:
            cid = client.cid

            # Initialize each client's local model with the current global model
            # if we haven't seen that client before.
            if cid not in self.client_local_models:
                self.client_local_models[cid] = [np.copy(arr) for arr in global_ndarrays]

            local_parameters = ndarrays_to_parameters(self.client_local_models[cid])

            config = {
                "round": server_round,
                "client_id": cid,
            }
            fit_configurations.append((client, FitIns(local_parameters, config)))

        self.last_fit_client_order = [client.cid for client in clients]
        return fit_configurations

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return self.global_parameters, {}

        # Store uploaded client models in the same order as results
        uploaded = []

        for client, fit_res in results:
            cid = client.cid
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            self.client_local_models[cid] = [np.copy(arr) for arr in ndarrays]
            uploaded.append((cid, ndarrays, fit_res.num_examples))


        # 1) Global aggregation (faithful to original servercross.py)
        aggregated_ndarrays = weighted_average_ndarrays(
            [(params, num_examples) for _, params, num_examples in uploaded]
        )
        self.global_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        # 2) Similarity + cross aggregation for the next round
        if server_round >= self.first_stage_bound:
            sim_tab = self.calculate_similarity(
                [params for _, params, _ in uploaded]
            )
            updated_locals = self.cross_aggregation(
                client_ids=[cid for cid, _, _ in uploaded],
                local_models=[params for _, params, _ in uploaded],
                sim_tab=sim_tab,
                iteration=server_round,
            )
            for cid, new_params in updated_locals.items():
                self.client_local_models[cid] = new_params
        else:
            # Before warmup threshold, reset participating clients to the new global model
            for cid, _, _ in uploaded:
                self.client_local_models[cid] = [np.copy(arr) for arr in aggregated_ndarrays]

        return self.global_parameters, {}

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ):
        clients = client_manager.sample(
            num_clients=self.min_evaluate_clients,
            min_num_clients=self.min_available_clients,
        )

        # Evaluate the GLOBAL model, matching original FedCross behavior
        eval_configurations = []
        for client in clients:
            config = {"round": server_round}
            eval_configurations.append((client, fl.common.EvaluateIns(self.global_parameters, config)))
        return eval_configurations

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        losses = []
        total_examples = 0

        for client, eval_res in results:
            losses.append((eval_res.loss, eval_res.num_examples))
            total_examples += eval_res.num_examples
        weighted_loss = sum(loss * n for loss, n in losses) / total_examples
        return float(weighted_loss), {}

    def evaluate(self, server_round: int, parameters: Parameters):
        # No centralized server-side evaluation
        return None

    # ------------------------------------------------------------------
    # Faithful FedCross helpers
    # ------------------------------------------------------------------

    def calculate_similarity(self, local_models: List[List[np.ndarray]]) -> List[List[float]]:
        model_num = len(local_models)
        sim_tab = [[0.0 for _ in range(model_num)] for _ in range(model_num)]

        state_dicts = [
            ndarrays_to_torch_state_dict(self.model_keys, nds) for nds in local_models
        ]

        for k in range(model_num):
            for j in range(k):
                s = 0.0
                cnt = 0
                sub_a = None
                sub_b = None

                for key in state_dicts[k].keys():
                    a = state_dicts[k][key].view(-1).float().cpu()
                    b = state_dicts[j][key].view(-1).float().cpu()

                    if cnt % 2 == 0:
                        sub_a = a
                        sub_b = b
                    else:
                        sub_a = torch.cat((sub_a, a), dim=0)
                        sub_b = torch.cat((sub_b, b), dim=0)

                    if cnt % 2 == 1:
                        s += F.cosine_similarity(sub_a, sub_b, dim=0).item()
                    cnt += 1

                if sub_a is not None and sub_b is not None:
                    s += F.cosine_similarity(sub_a, sub_b, dim=0).item()

                sim_tab[k][j] = s
                sim_tab[j][k] = s

        return sim_tab

    def aggregate_parameters_cross(
        self, models: List[List[np.ndarray]], weights: List[float]
    ) -> List[np.ndarray]:
        total_count = sum(weights)
        aggregated = []

        for layer_idx in range(len(models[0])):
            layer_sum = np.zeros_like(models[0][layer_idx], dtype=np.float64)
            for w, model in zip(weights, models):
                layer_sum += np.asarray(model[layer_idx], dtype=np.float64) * (w / total_count)
            aggregated.append(layer_sum.astype(models[0][layer_idx].dtype))

        return aggregated

    def cross_aggregation(
        self,
        client_ids: List[str],
        local_models: List[List[np.ndarray]],
        sim_tab: List[List[float]],
        iteration: int,
    ) -> Dict[str, List[np.ndarray]]:
        w_locals_num = len(local_models)
        updated = {}

        for j in range(w_locals_num):
            maxtag = 0
            submax = 1 if w_locals_num > 1 else 0
            mintag = (j + 1) % w_locals_num if w_locals_num > 1 else 0

            for p in range(w_locals_num):
                if sim_tab[j][p] > sim_tab[j][maxtag]:
                    submax = maxtag
                    maxtag = p
                elif p != maxtag and sim_tab[j][p] > sim_tab[j][submax]:
                    submax = p

                if p != j and sim_tab[j][p] < sim_tab[j][mintag]:
                    mintag = p

            rlist = []
            sub_list = []

            # self model
            rlist.append(self.cross_alpha)
            sub_list.append(local_models[j])

            if w_locals_num > 1:
                offset = iteration % (w_locals_num - 1) + 1

                if self.collaborative_model_select_strategy == 0:
                    k = (j + offset) % w_locals_num
                    rlist.append(1.0 - self.cross_alpha)
                    sub_list.append(local_models[k])
                elif self.collaborative_model_select_strategy == 1:
                    rlist.append(1.0 - self.cross_alpha)
                    sub_list.append(local_models[mintag])
                elif self.collaborative_model_select_strategy == 2:
                    rlist.append(1.0 - self.cross_alpha)
                    sub_list.append(local_models[maxtag])
                else:
                    raise ValueError(
                        f"Unsupported collaborative_model_select_strategy: "
                        f"{self.collaborative_model_select_strategy}"
                    )

            updated[client_ids[j]] = self.aggregate_parameters_cross(sub_list, rlist)

        return updated
