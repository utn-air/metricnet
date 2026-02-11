import os

import numpy as np
import torch
import yaml


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def from_numpy(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).float()


def load_data_stats() -> dict:
    with open(
        os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r"
    ) as f:
        data_config = yaml.safe_load(f)
    action_stats = {}
    for key in data_config["action_stats"]:
        action_stats[key] = np.array(data_config["action_stats"][key])
    return action_stats


ACTION_STATS = load_data_stats()


def normalize_data(data: np.ndarray, stats: dict) -> np.ndarray:
    ndata = (data - stats["min"]) / (stats["max"] - stats["min"])
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata: np.ndarray, stats: dict) -> np.ndarray:
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"]) + stats["min"]
    return data


def get_delta(actions: np.ndarray) -> np.ndarray:
    ex_actions = np.concatenate(
        [np.zeros((actions.shape[0], 1, actions.shape[-1])), actions], axis=1
    )
    delta = ex_actions[:, 1:] - ex_actions[:, :-1]
    return delta


def get_action(ndeltas, action_stats=ACTION_STATS) -> torch.Tensor:
    device = ndeltas.device
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    ndeltas = to_numpy(ndeltas)
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = np.cumsum(ndeltas, axis=1)
    return from_numpy(actions).to(device)
