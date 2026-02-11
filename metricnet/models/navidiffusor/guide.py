import os

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from depth_anything_v2.dpt import DepthAnythingV2

from .costmap_cfg import CostMapConfig
from .tsdf_cost_map import TsdfCostMap


def from_numpy(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).float()


class PathGuide:
    def __init__(self, device, ACTION_STATS, path_weights: str, guide_cfgs=None):
        """
        Parameters:
        """
        self.device = device
        self.guide_cfgs = guide_cfgs
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.l1_loss = nn.L1Loss(reduction="mean")
        self.robot_width = 0.6
        self.spatial_resolution = 0.1
        self.max_distance = 10
        self.bev_dist = self.max_distance / self.spatial_resolution
        self.delta_min = from_numpy(ACTION_STATS["min"]).to(self.device)
        self.delta_max = from_numpy(ACTION_STATS["max"]).to(self.device)

        # TODO: Pass in parameters instead of constants
        self.camera_intrinsics = np.array(
            [
                [364.8, 0, 640 / 2],
                [0, 364.8, 480 / 2],
                [0, 0, 1],
            ]
        )
        # robot to camera extrinsic
        self.camera_extrinsics = np.array(
            [[0, 0, 1, -0.000], [-1, 0, 0, -0.000], [0, -1, 0, -0.5], [0, 0, 0, 1]]
        )

        # depth anything v2 init
        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitg": {
                "encoder": "vitg",
                "features": 384,
                "out_channels": [1536, 1536, 1536, 1536],
            },
        }
        encoder = "vits"  # or 'vits', 'vitb', 'vitg'
        self.model = DepthAnythingV2(**model_configs[encoder])
        self.model.load_state_dict(
            torch.load(path_weights),
            map_location="cpu",
        )
        self.model = self.model.to(self.device).eval()

        # TSDF init
        self.tsdf_cfg = CostMapConfig()
        self.tsdf_cost_map = TsdfCostMap(
            self.tsdf_cfg.general, self.tsdf_cfg.tsdf_cost_map
        )

    def _norm_delta_to_ori_trajs(self, trajs):
        delta_tmp = (trajs + 1) / 2
        delta_ori = delta_tmp * (self.delta_max - self.delta_min) + self.delta_min
        trajs_ori = delta_ori.cumsum(dim=1)
        return trajs_ori

    def goal_cost(self, trajs, goal, scale_factor=None):
        trajs_ori = self._norm_delta_to_ori_trajs(trajs)
        if scale_factor is not None:
            trajs_ori *= scale_factor
        trajs_end_positions = trajs_ori[:, -1, :]

        distances = torch.norm(goal - trajs_end_positions, dim=1)

        gloss = 0.05 * torch.sum(distances)

        if trajs.grad is not None:
            trajs.grad.zero_()

        gloss.backward()
        return trajs.grad

    def generate_scale(self, n):
        scale = torch.linspace(0, 1, steps=n)

        squared_scale = scale**1

        return squared_scale.to(self.device)

    def depth_to_pcd(
        self,
        depth_image,
        camera_intrinsics,
        camera_extrinsics,
        resize_factor=1.0,
        height_threshold=0.5,
        max_distance=10.0,
    ):
        height, width = depth_image.shape
        # print("height: ", height, "width: ", width)
        fx, fy = (
            camera_intrinsics[0, 0] * resize_factor,
            camera_intrinsics[1, 1] * resize_factor,
        )
        cx, cy = (
            camera_intrinsics[0, 2] * resize_factor,
            camera_intrinsics[1, 2] * resize_factor,
        )

        x, y = np.meshgrid(np.arange(width), np.arange(height))
        z = depth_image.astype(np.float32)
        z_safe = np.where(z == 0, np.nan, z)
        z = 1 / z_safe
        x = (x - width / 2) * z / fx
        y = (y - height / 2) * z / fy
        non_ground_mask = (z > 0.5) & (z < max_distance)
        x_non_ground = x[non_ground_mask]
        y_non_ground = y[non_ground_mask]
        z_non_ground = z[non_ground_mask]

        points = np.stack((x_non_ground, y_non_ground, z_non_ground), axis=-1).reshape(
            -1, 3
        )

        extrinsics = camera_extrinsics
        homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
        transformed_points = (extrinsics @ homogeneous_points.T).T[:, :3]

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(transformed_points)

        return point_cloud

    def add_robot_dim(self, world_ps):
        tangent = world_ps[:, 1:, 0:2] - world_ps[:, :-1, 0:2]
        tangent = tangent / torch.norm(tangent, dim=2, keepdim=True)
        normals = tangent[:, :, [1, 0]] * torch.tensor(
            [-1, 1], dtype=torch.float32, device=world_ps.device
        )
        world_ps_inflated = torch.vstack([world_ps[:, :-1, :]] * 3)
        world_ps_inflated[:, :, 0:2] = torch.vstack(
            [
                world_ps[:, :-1, 0:2] + normals * self.robot_width / 2,
                world_ps[:, :-1, 0:2],  # center
                world_ps[:, :-1, 0:2] - normals * self.robot_width / 2,
            ]
        )
        return world_ps_inflated

    def get_cost_map_via_tsdf(self, img):
        original_width, original_height = img.size
        resize_factor = 0.25
        new_size = (
            int(original_width * resize_factor),
            int(original_height * resize_factor),
        )
        img = img.resize(new_size)
        depth_image = self.model.infer_image(
            cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        )
        pseudo_pcd = self.depth_to_pcd(
            depth_image,
            self.camera_intrinsics,
            self.camera_extrinsics,
            resize_factor=resize_factor,
        )

        self.tsdf_cost_map.LoadPointCloud(pseudo_pcd)
        data, coord = self.tsdf_cost_map.CreateTSDFMap()
        if data is None:
            self.cost_map = None
        else:
            self.cost_map = torch.tensor(data[0]).requires_grad_(False).to(self.device)

    def collision_cost(self, trajs, scale_factor=None):
        if self.cost_map is None:
            return torch.zeros(trajs.shape)
        batch_size, num_p, _ = trajs.shape
        trajs_ori = self._norm_delta_to_ori_trajs(trajs)
        trajs_ori = self.add_robot_dim(trajs_ori)
        if scale_factor is not None:
            trajs_ori *= scale_factor
        norm_inds, _ = self.tsdf_cost_map.Pos2Ind(trajs_ori)
        cost_grid = self.cost_map.T.expand(trajs_ori.shape[0], 1, -1, -1)
        oloss_M = (
            F.grid_sample(
                cost_grid,
                norm_inds[:, None, :, :],
                mode="bicubic",
                padding_mode="border",
                align_corners=False,
            )
            .squeeze(1)
            .squeeze(1)
        )
        oloss_M = oloss_M.to(torch.float32)

        loss = 0.003 * torch.sum(oloss_M, axis=1)
        if trajs.grad is not None:
            trajs.grad.zero_()
        loss.backward(torch.ones_like(loss))
        cost_list = loss[1::3]
        generate_scale = self.generate_scale(trajs.shape[1])
        return generate_scale.unsqueeze(1).unsqueeze(0) * trajs.grad, cost_list

    def get_gradient(
        self,
        trajs,
        alpha=0.3,
        t=None,
        goal_pos=None,
        ACTION_STATS=None,
        scale_factor=None,
    ):
        trajs_in = trajs.detach().requires_grad_(True).to(self.device)
        if goal_pos is not None:
            goal_pos = torch.tensor(goal_pos).to(self.device)
            goal_cost = self.goal_cost(trajs_in, goal_pos, scale_factor=scale_factor)
            cost = goal_cost
            return cost, None
        else:
            collision_cost, cost_list = self.collision_cost(
                trajs_in, scale_factor=scale_factor
            )
            cost = collision_cost
        return cost, cost_list
