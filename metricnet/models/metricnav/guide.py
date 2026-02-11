import importlib.util
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans

from .costmap_cfg import CostMapConfig
from .depth_anything_v2_metric.dpt import DepthAnythingV2
from .tsdf_cost_map import TsdfCostMap

# import matplotlib.pyplot as plt
# from termcolor import colored





def from_numpy(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).float()


def check_tensor(tensor, name="tensor"):
    if tensor.grad is not None:
        print(f"{name} grad: {tensor.grad}")
    else:
        print(f"{name} grad is None")


class PathGuide:
    def __init__(self, device, ACTION_STATS, path_weights: str, guide_cfgs=None):
        """
        Parameters:
        """
        self.device = device
        self.guide_cfgs = guide_cfgs
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.l1_loss = nn.L1Loss(reduction="mean")
        self.robot_width = 0.3
        self.spatial_resolution = 0.1
        self.max_distance = 10
        self.bev_dist = self.max_distance / self.spatial_resolution
        self.delta_min = from_numpy(ACTION_STATS["min"]).to(self.device)
        self.delta_max = from_numpy(ACTION_STATS["max"]).to(self.device)

        # TODO: Pass in parameters instead of constants
        # self.camera_intrinsics = np.array(
        #     [
        #         [607.99658203125, 0, 642.2532958984375],
        #         [0, 607.862060546875, 366.3480224609375],
        #         [0, 0, 1],
        #     ]
        # )

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
        encoder = "vitb"  # or 'vits', 'vitb', 'vitg'
        self.model = DepthAnythingV2(**{**model_configs[encoder], "max_depth": 20})
        # self.model = DepthAnythingV2(**model_configs[encoder])
        self.model.load_state_dict(
            torch.load(
                path_weights,
                map_location="cpu",
            )
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
        # import time

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
        scale = torch.linspace(0, 0.01, steps=n)
        # scale = torch.ones(n) * 0.0

        squared_scale = scale**1

        return squared_scale.to(self.device)

    # def depth_to_pcd(
    #     self,
    #     depth_image,
    #     camera_intrinsics,
    #     camera_extrinsics,
    #     resize_factor=1.0,
    #     height_threshold=0.5,
    #     max_distance=10.0,
    # ):
    #     height, width = depth_image.shape
    #     print("height: ", height, "width: ", width)
    #     fx, fy = (
    #         camera_intrinsics[0, 0] * resize_factor,
    #         camera_intrinsics[1, 1] * resize_factor,
    #     )
    #     cx, cy = (
    #         camera_intrinsics[0, 2] * resize_factor,
    #         camera_intrinsics[1, 2] * resize_factor,
    #     )

    #     x, y = np.meshgrid(np.arange(width), np.arange(height))
    #     z = depth_image.astype(np.float32)
    #     z_safe = np.where(z == 0, np.nan, z)
    #     z = 1 / z_safe
    #     x = (x - width / 2) * z / fx
    #     y = (y - height / 2) * z / fy
    #     non_ground_mask = (z > 0.5) & (z < max_distance)
    #     x_non_ground = x[non_ground_mask]
    #     y_non_ground = y[non_ground_mask]
    #     z_non_ground = z[non_ground_mask]

    #     points = np.stack((x_non_ground, y_non_ground, z_non_ground), axis=-1).reshape(
    #         -1, 3
    #     )

    #     extrinsics = camera_extrinsics
    #     homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    #     transformed_points = (extrinsics @ homogeneous_points.T).T[:, :3]

    #     point_cloud = o3d.geometry.PointCloud()
    #     point_cloud.points = o3d.utility.Vector3dVector(transformed_points)

    #     return point_cloud

    def combine_colored_pcds(self, pcd1, pcd2):
        # Assign colors: red, green, blue
        pcd1.paint_uniform_color([1, 0, 0])
        pcd2.paint_uniform_color([0, 0, 1])
        green_color = np.array([[0.0, 1.0, 0.0]])

        zero_point = np.array([[0.0, 0.0, 0.0]])
        combined_pcd = o3d.geometry.PointCloud()
        combined_pcd.points = o3d.utility.Vector3dVector(
            np.vstack((np.asarray(pcd1.points), np.asarray(pcd2.points), zero_point))
        )
        combined_pcd.colors = o3d.utility.Vector3dVector(
            np.vstack((np.asarray(pcd1.colors), np.asarray(pcd2.colors), green_color))
        )

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.5, origin=[0, 0, 0]
        )

        o3d.visualization.draw_geometries([combined_pcd, mesh_frame])
        return combined_pcd

    def remove_ground_plane(self, pcd):
        points_array = np.asarray(pcd.points)

        try:
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=0.2, ransac_n=3, num_iterations=1000
            )
            # outlier_cloud = pcd.select_by_index(inliers, invert=True)

            i = 0
            while np.argmax(plane_model[:-1]) != 2:
                i += 1
                pcd = pcd.select_by_index(inliers, invert=True)
                plane_model, inliers = pcd.segment_plane(
                    distance_threshold=0.2, ransac_n=3, num_iterations=10000
                )

        except:
            print("RANSAC failed, no plane found, all points are obstacles")
            inliers = []

        outliers_index = set(range(points_array.shape[0])) - set(inliers)
        outliers_index = list(outliers_index)

        inliers_index = list(inliers)

        no_ground_scan = points_array[outliers_index]
        ground_scan = points_array[inliers_index]

        pcd_obs = o3d.geometry.PointCloud()
        pcd_obs.points = o3d.utility.Vector3dVector(no_ground_scan)

        pcd_free = o3d.geometry.PointCloud()
        pcd_free.points = o3d.utility.Vector3dVector(ground_scan)

        # self.combine_colored_pcds(pcd_obs, pcd_free)

        return ground_scan, no_ground_scan

        # return no_ground_scan

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

        # Generate mesh grid and calculate point cloud coordinates
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = (x - width / 2) / fx
        y = (y - height / 2) / fy
        z = depth_image.astype(np.float32)
        # breakpoint()

        points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(
            -1, 3
        )
        z = z.reshape(-1)
        valid_points = points[(z > 0.01) & (z < max_distance)]

        extrinsics = camera_extrinsics
        transformed_points = (extrinsics[:3, :3] @ valid_points.T).T[:, :3]

        # Create the point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(transformed_points)

        free_points, obs_points = self.remove_ground_plane(pcd)

        # o3d.visualization.draw_geometries([pcd])

        return free_points, obs_points  # pcd

    def add_robot_dim(self, world_ps):
        tangent = world_ps[:, 1:, 0:2] - world_ps[:, :-1, 0:2]
        tangent = tangent / torch.norm(tangent, dim=2, keepdim=True)
        normals = tangent[:, :, [1, 0]] * torch.tensor(
            [-1, 1], dtype=torch.float16, device=world_ps.device
        )
        world_ps_inflated = torch.vstack([world_ps[:, :-1, :]] * 3)
        world_ps_inflated[:, :, 0:2] = torch.vstack(
            [
                world_ps[:, :-1, 0:2] + normals * self.robot_width / 2,
                world_ps[:, :-1, 0:2],  # center
                world_ps[:, :-1, 0:2] - normals * self.robot_width / 2,
            ]
        )

        del normals, tangent

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

        free_points, obs_points = self.depth_to_pcd(
            depth_image,
            self.camera_intrinsics,
            self.camera_extrinsics,
            resize_factor=resize_factor,
            max_distance=20.0,
        )

        # import open3d as o3d
        # o3d.visualization.draw_geometries([pseudo_pcd])

        self.tsdf_cost_map.LoadPointCloudNew(free_p=free_points, obs_p=obs_points)
        data, coord = self.tsdf_cost_map.CreateTSDFMap()

        # vmin= 0.0
        # vmax = 5.0
        # plt.cla()
        # flipped = np.flip(np.flip(data[0], axis=1), axis=0)
        # plt.imshow(flipped, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        # plt.colorbar(label='values')
        # plt.savefig("tensor_grid_plot.png", dpi=300, bbox_inches="tight")
        # plt.close()

        # plt.imshow(depth_image)
        # plt.axis('off')
        # plt.savefig("depth_image.png", dpi=300, bbox_inches="tight")
        # plt.close()

        # plt.imshow(img)
        # plt.axis('off')
        # plt.savefig("input_image.png", dpi=300, bbox_inches="tight")
        # plt.close()

        if data is None:
            self.cost_map = None
        else:
            self.cost_map = torch.tensor(data[0]).requires_grad_(False).to(self.device)

        # del depth_image, obs_points, free_points, data, coord

    def collision_cost(self, trajs, scale_factor=None, k=None):
        if self.cost_map is None:
            return torch.zeros(trajs.shape)

        batch_size, num_p, _ = trajs.shape
        trajs_ori = self._norm_delta_to_ori_trajs(trajs)
        trajs_ori = self.add_robot_dim(trajs_ori)
        if scale_factor is not None:
            trajs_ori = trajs_ori * scale_factor[:, None, None]

        norm_inds, _ = self.tsdf_cost_map.Pos2Ind(trajs_ori)
        cost_grid = self.cost_map.T.expand(trajs_ori.shape[0], 1, -1, -1)
        oloss_M = (
            F.grid_sample(
                cost_grid,
                norm_inds[:, None, :, :],
                mode="bicubic",
                padding_mode="zeros",  # TODO
                align_corners=False,
            )
            .squeeze(1)
            .squeeze(1)
        )
        oloss_M = torch.where(torch.abs(oloss_M) < 1e-3, -10.0, oloss_M)
        # oloss_M = oloss_M.to(torch.float32)

        loss = -0.003 * torch.sum(oloss_M, axis=1)
        if trajs.grad is not None:
            trajs.grad.zero_()
        # if scale_factor.grad is not None:
        #     scale_factor.grad.zero_()
        loss.backward(torch.ones_like(loss))
        cost_list = loss[1::3]
        generate_scale = self.generate_scale(trajs.shape[1])

        oloss_M = oloss_M.detach().cpu().numpy()

        del loss, trajs_ori, norm_inds, cost_grid

        return (
            generate_scale.unsqueeze(1).unsqueeze(0) * trajs.grad,
            0.1 * 0.01,
            cost_list,
            np.sum(oloss_M, axis=1),
        )

    def collision_cost_checker(self, norm_inds):
        norm_inds = norm_inds.to(self.device)
        cost_grid = self.cost_map.T.expand(norm_inds.shape[0], 1, -1, -1)
        oloss_M = (
            F.grid_sample(
                cost_grid,
                norm_inds[:, None, :, :],
                mode="bicubic",
                padding_mode="zeros",  # TODO
                align_corners=False,
            )
            .squeeze(1)
            .squeeze(1)
        )
        # oloss_M = oloss_M.to(torch.float32)
        costs = torch.sum(oloss_M, axis=1)

        del norm_inds, cost_grid, oloss_M

        return costs

    def get_gradient(
        self,
        trajs,
        alpha=0.3,
        t=None,
        goal_pos=None,
        ACTION_STATS=None,
        scale_factor=None,
        k=None,
    ):
        trajs_in = trajs.detach().requires_grad_(True).to(self.device)
        # scale_factor_in = scale_factor.detach().requires_grad_(True).to(self.device)
        scale_factor_in = scale_factor
        if goal_pos is not None:
            goal_pos = torch.tensor(goal_pos).to(self.device)
            goal_cost = self.goal_cost(trajs_in, goal_pos, scale_factor=scale_factor_in)
            cost = goal_cost
            return cost, None
        else:
            collision_cost, scale_grad, cost_list, costs = self.collision_cost(
                trajs_in, scale_factor=scale_factor_in, k=k
            )
            cost = collision_cost

        del trajs_in, scale_factor_in

        return cost, scale_grad, cost_list, costs

    def get_gradient_goal_collision(
        self,
        trajs,
        goal_pos,
        alpha=0,
        beta=0,
        t=None,
        ACTION_STATS=None,
        scale_factor=None,
        k=None,
    ):
        trajs_in = trajs.detach().requires_grad_(True).to(self.device)
        goal_pos_in = (
            torch.tensor(goal_pos).detach().requires_grad_(False).to(self.device)
        )
        scale_factor = scale_factor.detach().requires_grad_(False)

        trajs_ori = self._norm_delta_to_ori_trajs(trajs_in)
        trajs_ori = self.add_robot_dim(trajs_ori)
        if scale_factor is not None:
            trajs_ori *= (
                torch.repeat_interleave(scale_factor, 3).unsqueeze(1).unsqueeze(1)
            )

        # Collision Cost
        if self.cost_map is None:
            collision_loss = torch.zeros(trajs_in.shape[0]).to(self.device)
        else:
            n_inds, _ = self.tsdf_cost_map.Pos2Ind(trajs_ori)
            cost_grid = self.cost_map.T.expand(trajs_ori.shape[0], 1, -1, -1)
            oloss_M = (
                F.grid_sample(
                    cost_grid,
                    n_inds[:, None, :, :],
                    mode="bicubic",
                    padding_mode="zeros",  # TODO
                    align_corners=False,
                )
                .squeeze(1)
                .squeeze(1)
            )

            collision_loss = -torch.sum(oloss_M, axis=1)

        # Goal Cost
        trajs_end_positions = trajs_ori[1::3, 5, :]

        cosine_sim = F.cosine_similarity(goal_pos_in, trajs_end_positions, dim=1)

        # goal_loss = 0.05 * torch.norm(goal_pos - trajs_end_positions, dim=1)
        goal_loss = 1 - cosine_sim

        if trajs_in.grad is not None:
            trajs_in.grad.zero_()

        total_loss = (
            alpha * torch.repeat_interleave(goal_loss, 3) + beta * collision_loss
        )
        total_loss.backward(torch.ones_like(total_loss))

        generate_scale = self.generate_collision_goal_scale(trajs_in.shape[1])

        return generate_scale.unsqueeze(1).unsqueeze(0) * trajs_in.grad

    def point_costs(self, norm_inds):
        norm_inds = norm_inds.to(self.device)
        cost_grid = self.cost_map.T.expand(norm_inds.shape[0], 1, -1, -1)
        oloss_M = (
            F.grid_sample(
                cost_grid,
                norm_inds[:, None, :, :],
                mode="bicubic",
                padding_mode="zeros",  # TODO
                align_corners=False,
            )
            .squeeze(1)
            .squeeze(1)
        )

        return oloss_M

    def generate_collision_goal_scale(self, n):
        # scale = torch.linspace(0, 0.1, steps=n)
        scale = torch.ones(n) * 1e-1

        squared_scale = scale**1

        return squared_scale.to(self.device)

    def plot_with_cost_map(self, series, palette, path):
        H, W = self.cost_map.shape[:2]

        def xy_from_inds(inds):
            # inds: (..., 2) with [row, col]
            x = W - inds[..., 1]
            y = H - inds[..., 0]
            return x, y

        flipped = np.flip(np.flip(self.cost_map.detach().cpu().numpy(), axis=1), axis=0)
        plt.imshow(flipped, cmap="viridis", aspect="auto")

        for (label, arr, style), color in zip(series, palette):
            for i in range(arr.shape[0]):
                x, y = xy_from_inds(arr[i])  # arr[i] is (8, 2)
                plt.plot(x, y, label=label if i == 0 else None, color=color, **style)

        plt.legend()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

    def choose_goal_spherical(self, goal_positions):
        # Goal positions has shape [N, 2]

        # --- No changes needed in this part ---
        n_ids, _ = self.tsdf_cost_map.Pos2Ind(torch.tensor(goal_positions))
        g_cost = self.point_costs(n_ids).detach().cpu().numpy()
        best_indices = np.arange(g_cost.shape[1])

        # Isolate the relevant goal positions for clustering
        active_goals = goal_positions[best_indices]

        # 1. NORMALIZE VECTORS TO UNIT LENGTH
        # This is the core step for spherical k-means. It projects points onto a unit circle.
        # We add a small epsilon to avoid division by zero for a potential [0, 0] vector.
        norms = np.linalg.norm(active_goals, axis=1, keepdims=True)
        goals_norm = active_goals / (norms + 1e-8)

        # --- Perform KMeans on the normalized data ---
        k = 3
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init="auto", random_state=42)
        labels = kmeans.fit_predict(goals_norm)

        # Cluster centers are also vectors (directions)
        centers = kmeans.cluster_centers_

        # 2. NORMALIZE THE CLUSTER CENTERS
        # The centroids from sklearn are arithmetic means, so they might not be on the unit circle.
        # We normalize them to represent a pure direction.
        centers_norm = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)

        # --- Choose the cluster center with the most points (same logic) ---
        unique, counts = np.unique(labels, return_counts=True)
        most_common_cluster_idx = np.argmax(counts)

        # Get the normalized center of the most popular cluster
        chosen_center = centers_norm[most_common_cluster_idx]

        # 3. FIND THE CLOSEST POINT USING COSINE SIMILARITY (DOT PRODUCT)
        # Instead of Euclidean distance, we find the original point whose direction
        # is most similar to the chosen cluster's direction.
        # For unit vectors, the dot product is the cosine of the angle between them.
        # A higher dot product means a smaller angle.

        # We calculate the similarity with all original (normalized) points
        all_goals_norm = goal_positions / (
            np.linalg.norm(goal_positions, axis=1, keepdims=True) + 1e-8
        )
        similarities = np.dot(all_goals_norm, chosen_center)

        # The best goal is the one with the highest cosine similarity (smallest angle)
        best_goal_idx = np.argmax(similarities)

        return best_goal_idx, labels
