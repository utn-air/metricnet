import os
from typing import Any, Dict, List

import click
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import yaml
from PIL import Image as PILImage
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torchvision import transforms

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from metricnet.models.metricnet.metricnet import MetricNetBatched as MetricNet
from metricnet.models.nomad.nomad import DenseNetwork, NoMaD
from metricnet.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn

IMAGE_ASPECT_RATIO = 4 / 3  # width / height


def load_config(filepath: str) -> Dict[str, Any]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file {filepath} not found.")
    with open(filepath, "r") as file:
        return yaml.safe_load(file)


def load_model(
    model_path: str,
    config: Dict[str, Any],
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    model_type = config["model_type"]

    if config["model_type"] == "nomad":
        if config["vision_encoder"] == "nomad_vint":
            assert config.get("include_observation_in_goal_encoder", True), (
                "include_observation_in_goal_encoder should be True, DO NOT USE!"
            )
            assert not config.get("include_action_in_state_decoder", False), (
                "include_action_in_state_decoder should be False, DO NOT USE!"
            )
            vision_encoder = NoMaD_ViNT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
                depth_cfg=config["depth"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        else:
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")

        noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=config["encoding_size"],
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        )
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if model_type == "nomad":
        state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        loaded_model = checkpoint["model"]
        try:
            state_dict = loaded_model.module.state_dict()
            model.load_state_dict(state_dict, strict=False)
        except AttributeError as _:
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict, strict=False)
    model.to(device)

    click.echo(
        click.style(f">> Loaded {model_type} model from {model_path}", fg="green")
    )

    return model


def get_metricNet(config: Dict[str, Any], device: torch.device) -> MetricNet:
    metricNet = MetricNet(
        d_model=config["scale"]["d_model"],
        n_conv_res_blocks=config["scale"].get("n_conv_res_blocks", 1),
        n_lin_res_blocks=config["scale"].get("n_lin_res_blocks", 1),
    )
    if config.get("scale", {}).get("enabled", False):
        ckpt_path = config["scale"]["weights"]
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        consume_prefix_in_state_dict_if_present(ckpt, "_orig_mod.")
        metricNet.load_state_dict(ckpt, strict=True)
    metricNet.to(device)
    return metricNet


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.cpu().detach().numpy()


def transform_images_nomad(
    pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False
) -> torch.Tensor:
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if not isinstance(pil_imgs, list):
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_img = pil_img.resize(image_size)
        transf_img = transform_type(pil_img)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)


def transform_obs_metricnet(
    pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False
) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if not isinstance(pil_imgs, list):
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(
                    pil_img, (h, int(h * IMAGE_ASPECT_RATIO))
                )  # crop to the right ratio
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_img = pil_img.resize(image_size)
        transf_img = transform_type(pil_img)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=0)
