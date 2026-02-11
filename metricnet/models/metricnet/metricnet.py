from collections import OrderedDict

import torch
import torch.nn as nn
from depth_anything_v2.dinov2 import DINOv2
from efficientnet_pytorch import EfficientNet

from .utils import Conv2dResidualBlock, LinearResidualBlock, PositionalEncoding


class MetricNet(nn.Module):
    def __init__(
        self,
        d_model: int = 384,
        n_conv_res_blocks: int = 1,
        n_lin_res_blocks: int = 1,
    ):
        super(MetricNet, self).__init__()
        self.depth_encoder = DINOv2(model_name="vits")
        self.depth_project = nn.Sequential(
            OrderedDict(
                {
                    str(i): LinearResidualBlock(d=384, expansion=4, drop=0.1)
                    for i in range(n_lin_res_blocks)
                }
            )
        )
        self.waypoint_encoder = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(approximate="tanh"),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(approximate="tanh"),
            nn.Flatten(),
            nn.Linear(128 * 2, d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(approximate="tanh"),
        )
        self.image_encoder = EfficientNet.from_pretrained(
            "efficientnet-b0", in_channels=3, num_classes=d_model
        )
        self.img_proj = nn.Sequential(
            OrderedDict(
                {
                    str(i): Conv2dResidualBlock(d_in=1280, d=d_model, k=5)
                    for i in range(n_conv_res_blocks)
                }
            )
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=6,
                dim_feedforward=d_model * 4,
                activation="gelu",
                batch_first=True,
                dropout=0.1,
            ),
            num_layers=3,
        )
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len=307)
        self.scale_pred_net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(approximate="tanh"),
            nn.Linear(128, 64),
            nn.GELU(approximate="tanh"),
            nn.Linear(64, 1),
        )

    def forward(self, obs_img: torch.Tensor, waypoint: torch.Tensor) -> torch.Tensor:
        # retrieve features
        img_features = self.img_proj(self.image_encoder.extract_features(obs_img))
        waypoint_features = self.waypoint_encoder(waypoint.permute(0, 2, 1)).unsqueeze(
            1
        )
        depth_features = self.depth_project(
            self.depth_encoder.get_intermediate_layers(
                obs_img, n=1, return_class_token=False
            )[0]
        )
        # CLS token
        cls_token = self.cls_token.expand(img_features.size(0), 1, -1)
        # get tokens
        tokens = self.positional_encoding(
            torch.cat(
                (cls_token, img_features, depth_features, waypoint_features), dim=1
            )
        )
        # transformer
        transformer_output = self.transformer_encoder(tokens)
        scale_output = self.scale_pred_net(transformer_output[:, 0]).squeeze()
        return scale_output


class MetricNetBatched(MetricNet):
    def forward(
        self,
        obs_img: torch.Tensor,
        waypoint: torch.Tensor,
        repeat_features: bool = False,
    ):
        # retrieve features
        img_features = self.img_proj(self.image_encoder.extract_features(obs_img))
        waypoint_features = self.waypoint_encoder(waypoint.permute(0, 2, 1)).unsqueeze(
            1
        )
        depth_features = self.depth_project(
            self.depth_encoder.get_intermediate_layers(
                obs_img, n=1, return_class_token=False
            )[0]
        )
        # CLS token
        cls_token = self.cls_token.expand(waypoint_features.size(0), 1, -1)
        # get tokens
        if repeat_features:
            tokens = self.positional_encoding(
                torch.cat(
                    (
                        cls_token,
                        torch.repeat_interleave(
                            img_features,
                            waypoint.shape[0] // img_features.shape[0],
                            dim=0,
                        ),
                        torch.repeat_interleave(
                            depth_features,
                            waypoint.shape[0] // depth_features.shape[0],
                            dim=0,
                        ),
                        waypoint_features,
                    ),
                    dim=1,
                )
            )
        else:
            tokens = self.positional_encoding(
                torch.cat(
                    (cls_token, img_features, depth_features, waypoint_features), dim=1
                )
            )
        # transformer
        transformer_output = self.transformer_encoder(tokens)
        scale_output = self.scale_pred_net(transformer_output[:, 0]).squeeze()
        return scale_output
