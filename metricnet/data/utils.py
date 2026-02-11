import os
from typing import Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

VISUALIZATION_IMAGE_SIZE = (160, 120)
IMAGE_ASPECT_RATIO = 4 / 3


def get_data_path(
    data_folder: str,
    f: str,
    time: int,
    data_type: str = "image",
    drop_folder: bool = False,
) -> str:
    data_dict = {
        "image": ".jpg",
        "depth": "_depth.jpg",
        # add more data types here
    }
    if drop_folder:
        return os.path.join(f, f"{str(time)}{data_dict[data_type]}")
    return os.path.join(data_folder, f, f"{str(time)}{data_dict[data_type]}")


def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )


def to_local_coords(
    positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: Union[float, np.ndarray]
) -> np.ndarray:
    if isinstance(curr_yaw, np.ndarray):
        curr_yaw = curr_yaw[0]
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError("positions must be of shape (..., 2) or (..., 3)")
    return (positions - curr_pos).dot(rotmat)


def resize_and_aspect_crop(
    img: Image.Image,
    image_resize_size: Tuple[int, int],
    aspect_ratio: float = IMAGE_ASPECT_RATIO,
) -> torch.Tensor:
    w, h = img.size
    if w > h:
        img = TF.center_crop(img, (h, int(h * aspect_ratio)))
    else:
        img = TF.center_crop(img, (int(w / aspect_ratio), w))
    img = img.resize(image_resize_size)
    resize_img = TF.to_tensor(img)
    return resize_img
