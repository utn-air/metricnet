import io
import os
import pickle
from bisect import bisect_right
from typing import Optional, Tuple

import click
import lmdb
import numpy as np
import torch
import tqdm
import yaml
from PIL import Image

from metricnet.data.utils import get_data_path, resize_and_aspect_crop, to_local_coords


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        waypoint_spacing: list[int],
        len_traj_pred: int,
    ):
        # check waypoint_spacing is a list of positive integers
        assert 0 not in waypoint_spacing, (
            "waypoint_spacing should not contain 0, "
            "it is used to compute the actions and should be a positive integer"
        )

        # substitute environment variables in data_folder and data_split_folder
        self.data_folder = os.path.expandvars(data_folder)
        self.data_split_folder = os.path.expandvars(data_split_folder)
        self.dataset_name = dataset_name

        # load trajectory names
        traj_filename = os.path.join(self.data_split_folder, "traj_names.txt")
        with open(traj_filename, "r") as f:
            self.traj_names = f.read().split("\n")
        if "" in self.traj_names:  # remove empty strings
            self.traj_names.remove("")

        # load parameters
        self.image_size = image_size
        self.waypoint_spacing = sorted(waypoint_spacing)
        self.len_traj_pred = len_traj_pred

        # load data/data_config.yaml
        with open(
            os.path.join(os.path.dirname(__file__), "data_config.yaml"), "r"
        ) as f:
            all_data_config = yaml.safe_load(f)
        assert self.dataset_name in all_data_config, (
            f"Dataset {self.dataset_name} not found in data_config.yaml"
        )
        dataset_names = list(all_data_config.keys())
        dataset_names.sort()

        # use this index to retrieve the dataset name from the data_config.yaml
        self.dataset_index = dataset_names.index(self.dataset_name)
        self.data_config = all_data_config[self.dataset_name]
        self.trajectory_cache = {}
        self._load_index()
        self._build_caches()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_image_cache"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._build_caches()

    def _build_caches(self):
        cache_filename = os.path.expandvars(
            os.path.join(
                self.data_split_folder,
                f"dataset_{self.dataset_name}.lmdb",
            )
        )
        if not os.path.exists(os.path.dirname(cache_filename)):
            os.makedirs(os.path.dirname(cache_filename), exist_ok=True)

        # load all the trajectories into memory. These should already be loaded, but just in case.
        for traj_name in tqdm.tqdm(
            self.traj_names,
            dynamic_ncols=True,
            desc="Loading trajectories",
            colour="yellow",
        ):
            self._get_trajectory(traj_name)

        # if the cache file doesn't exist, create it
        if not os.path.exists(cache_filename):
            tqdm_iterator = tqdm.tqdm(
                self.index_to_data,
                dynamic_ncols=True,
                desc=f"Building LMDB cache for {self.dataset_name}",
            )
            with lmdb.open(cache_filename, map_size=2**40) as image_cache:
                with image_cache.begin(write=True) as txn:
                    for i, (traj_name, time, _) in enumerate(tqdm_iterator):
                        image_path = get_data_path(self.data_folder, traj_name, time)
                        image_name = get_data_path(
                            self.data_folder, traj_name, time, drop_folder=True
                        )
                        assert os.path.exists(image_path), (
                            f"Image {image_path} does not exist!"
                        )
                        with open(image_path, "rb") as f:
                            image_data = f.read()
                            assert image_data is not None and len(image_data) > 0, (
                                f"Image {image_path} is empty or could not be read!"
                            )
                            txn.put(image_name.encode(), image_data)
            click.echo(
                click.style(
                    f">> Built LMDB cache with {len(tqdm_iterator)} entries", fg="green"
                )
            )

        # reopen the cache file in read-only mode
        self._image_cache: lmdb.Environment = lmdb.open(
            cache_filename, readonly=True, max_readers=512
        )

    def _build_index(self) -> list[Tuple[str, int, int]]:
        # build the index to map from dataset index to (trajectory name, time, max_waypoint_distance)
        # initialize empty list
        samples = []

        # iterate over all trajectories
        for traj_name in tqdm.tqdm(
            self.traj_names,
            dynamic_ncols=True,
            desc="Building index",
        ):
            # load trajectory data
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])
            end_time = traj_len - self.len_traj_pred * min(self.waypoint_spacing)
            # ensure end_time is positive
            try:
                assert end_time > 0, (
                    f"Trajectory {traj_name} is too short for len_traj_pred {self.len_traj_pred} with waypoint spacing {min(self.waypoint_spacing)}."
                )
            except AssertionError:
                click.echo(
                    click.style(f"WARNING: {traj_name} is too short", fg="yellow")
                )
            # iterate over all possible starting times
            for curr_time in range(end_time):
                max_waypoint_distance = (traj_len - curr_time) // self.len_traj_pred
                samples.append((traj_name, curr_time, max_waypoint_distance))
        click.echo(
            click.style(f">> Built index with {len(samples)} samples", fg="green")
        )
        return samples

    def _load_index(self) -> None:
        index_to_data_path = os.path.join(
            self.data_split_folder,
            "dataset.pkl",
        )
        try:
            # load the index_to_data if it already exists (to save time)
            with open(index_to_data_path, "rb") as f:
                self.index_to_data = pickle.load(f)
        except FileNotFoundError:
            # if the index_to_data file doesn't exist, create it
            self.index_to_data = self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump(self.index_to_data, f)

    def _load_image(
        self, trajectory_name: str, time: int, data_type: str = "image"
    ) -> Optional[Image.Image]:
        # load image from LMDB cache
        image_path = get_data_path(
            self.data_folder,
            trajectory_name,
            time,
            data_type=data_type,
            drop_folder=True,
        )

        try:
            with self._image_cache.begin() as txn:
                image_buffer = txn.get(image_path.encode())
                image_bytes = bytes(image_buffer)
            img = resize_and_aspect_crop(
                img=Image.open(io.BytesIO(image_bytes)),
                image_resize_size=(224, 224),
                aspect_ratio=4 / 3,
            )
            return img
        except TypeError:
            breakpoint()
            click.echo(
                click.style(f"ERROR: Failed to load image {image_path}", fg="red")
            )

    def _compute_actions(
        self, traj_data: dict, curr_time: int, idx_of_waypoint_spacing: int
    ) -> Tuple[torch.Tensor, float]:
        # get future positions and yaws
        start_index = curr_time
        end_index = (
            curr_time
            + self.len_traj_pred * self.waypoint_spacing[idx_of_waypoint_spacing]
            + 1
        )
        yaw = traj_data["yaw"][
            start_index : end_index : self.waypoint_spacing[idx_of_waypoint_spacing]
        ]
        positions = traj_data["position"][
            start_index : end_index : self.waypoint_spacing[idx_of_waypoint_spacing]
        ]

        # ensure yaw is 1D array
        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)
        if yaw.shape != (self.len_traj_pred + 1,):
            const_len = self.len_traj_pred + 1 - yaw.shape[0]
            yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            positions = np.concatenate(
                [positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0
            )

        assert yaw.shape == (self.len_traj_pred + 1,), (
            f"{yaw.shape} and {(self.len_traj_pred + 1,)} should be equal"
        )

        # ensure positions is 2D array
        assert positions.shape == (self.len_traj_pred + 1, 2), (
            f"{positions.shape} and {(self.len_traj_pred + 1, 2)} should be equal"
        )

        # convert positions to local coordinates
        waypoints = to_local_coords(positions, positions[0], yaw[0])
        actions = waypoints[1:]

        # get mean distance between waypoints (shape 8,2)
        mean_wp_distance = np.mean(
            np.linalg.norm(np.diff(waypoints.astype(float), axis=0), axis=1)
        )
        actions[:, :2] /= mean_wp_distance

        # ensure actions shape is (len_traj_pred, 2)
        assert actions.shape == (self.len_traj_pred, 2), (
            f"{actions.shape} and {(self.len_traj_pred, 2)} should be equal"
        )

        return actions, mean_wp_distance

    def _get_trajectory(self, trajectory_name: str) -> dict:
        # check if trajectory is already in cache
        if trajectory_name in self.trajectory_cache:
            return self.trajectory_cache[trajectory_name]
        else:
            with open(
                os.path.join(self.data_folder, trajectory_name, "traj_data.pkl"), "rb"
            ) as f:
                traj_data = pickle.load(f)
            self.trajectory_cache[trajectory_name] = traj_data
            return traj_data

    def __len__(self) -> int:
        return len(self.index_to_data)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f_curr, curr_time, max_waypoint_distance = self.index_to_data[i]

        # load other trajectory data
        curr_traj_data = self._get_trajectory(f_curr)
        curr_traj_len = len(curr_traj_data["position"])
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

        # get maximum waypoint spacing
        if max_waypoint_distance > min(self.waypoint_spacing):
            idx_of_waypoint_spacing = torch.randint(
                0, bisect_right(self.waypoint_spacing, max_waypoint_distance), (1,)
            ).item()
        else:
            idx_of_waypoint_spacing = 0

        # load image
        obs_image = self._load_image(f_curr, curr_time, data_type="image")

        # compute actions
        actions, mean_wp_distance = self._compute_actions(
            curr_traj_data, curr_time, idx_of_waypoint_spacing
        )

        actions_torch = torch.as_tensor(
            actions.astype(dtype=np.float32), dtype=torch.float32
        )
        return (
            torch.as_tensor(obs_image, dtype=torch.float32),
            torch.as_tensor(
                mean_wp_distance,
                dtype=torch.float32,
            ),
            actions_torch,
        )
