# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
# python
import os

import numpy as np
import open3d as o3d
import torch
from scipy import ndimage
from scipy.ndimage import gaussian_filter

# imperative-cost-map
from .costmap_cfg import GeneralCostMapConfig, TsdfCostMapConfig

# import matplotlib.pyplot as plt



class TsdfCostMap:
    """
    Cost Map based on geometric information
    """

    def __init__(self, cfg_general: GeneralCostMapConfig, cfg_tsdf: TsdfCostMapConfig):
        self._cfg_general = cfg_general
        self._cfg_tsdf = cfg_tsdf
        # set init flag
        self.is_map_ready = False
        # init point clouds
        self.obs_pcd = o3d.geometry.PointCloud()
        self.free_pcd = o3d.geometry.PointCloud()
        return

    def Pos2Ind(self, points):
        start_xy = torch.tensor(
            [self.start_x, self.start_y], dtype=torch.float64, device=points.device
        ).expand(1, 1, -1)
        H = (points - start_xy) / self._cfg_general.resolution
        # mask = torch.logical_and(
        #     (H > 0).all(axis=2),
        #     (
        #         H
        #         < torch.tensor([self.num_x, self.num_y], device=points.device)[
        #             None, None, :
        #         ]
        #     ).all(axis=2),
        # )
        return self.NormInds(H), H

    def NormInds(self, H):
        norm_matrix = torch.tensor(
            [self.num_x / 2.0, self.num_y / 2.0], dtype=torch.float64, device=H.device
        )
        H = (H - norm_matrix) / norm_matrix
        return H

    def UpdatePCDwithPs(self, P_obs, P_free, is_downsample=False):
        self.obs_pcd.points = o3d.utility.Vector3dVector(P_obs)
        self.free_pcd.points = o3d.utility.Vector3dVector(P_free)
        if is_downsample:
            self.obs_pcd = self.obs_pcd.voxel_down_sample(self._cfg_general.resolution)
            self.free_pcd = self.free_pcd.voxel_down_sample(
                self._cfg_general.resolution * 0.85
            )

        self.obs_points = np.asarray(self.obs_pcd.points)
        self.free_points = np.asarray(self.free_pcd.points)
        # print(
        #     "number of obs points: %d, free points: %d"
        #     % (self.obs_points.shape[0], self.free_points.shape[0])
        # )

    def ReadPointFromFile(self):
        pcd_load = o3d.io.read_point_cloud(
            os.path.join(self._cfg_general.root_path, self._cfg_general.ply_file)
        )
        obs_p, free_p = self.TerrainAnalysis(np.asarray(pcd_load.points))
        self.UpdatePCDwithPs(obs_p, free_p, is_downsample=True)
        if self._cfg_tsdf.filter_outliers:
            obs_p = self.FilterCloud(self.obs_points)
            free_p = self.FilterCloud(self.free_points, outlier_filter=False)
            self.UpdatePCDwithPs(obs_p, free_p)
        self.UpdateMapParams()
        return

    def LoadPointCloud(self, pcd):
        obs_p, free_p = self.TerrainAnalysis(np.asarray(pcd.points))

        # o3d.visualization.draw_geometries([pcd])    
        # o3d.io.write_point_cloud("pcd.ply", pcd)

        # Combine three point clouds with different colors
        def combine_colored_pcds(pcd1, pcd2):
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

            o3d.visualization.draw_geometries([combined_pcd])
            return combined_pcd

        # Create the point cloud
        # pcd_obs = o3d.geometry.PointCloud()
        # pcd_obs.points = o3d.utility.Vector3dVector(obs_p)
        # # o3d.visualization.draw_geometries([pcd_obs])
        # o3d.io.write_point_cloud("obs_pcd.ply", pcd_obs)

        # pcd_free = o3d.geometry.PointCloud()
        # pcd_free.points = o3d.utility.Vector3dVector(free_p)
        # # o3d.visualization.draw_geometries([pcd_free])
        # o3d.io.write_point_cloud("free_pcd.ply", pcd_free)

        # combine_colored_pcds(pcd_obs, pcd_free)

        self.UpdatePCDwithPs(obs_p, free_p, is_downsample=True)
        if self._cfg_tsdf.filter_outliers:
            obs_p = self.FilterCloud(self.obs_points, outlier_filter=False)
            free_p = self.FilterCloud(self.free_points, outlier_filter=False)
            self.UpdatePCDwithPs(obs_p, free_p)
        self.UpdateMapParams()
        return


    def LoadPointCloudNew(self, free_p, obs_p):
        obs_p = self.TerrainAnalysisNew(obs_p)

        # o3d.visualization.draw_geometries([pcd])    
        # o3d.io.write_point_cloud("pcd.ply", pcd)

        # Combine three point clouds with different colors
        def combine_colored_pcds(pcd1, pcd2):
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

            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5, origin=[0, 0, 0])

            o3d.visualization.draw_geometries([combined_pcd, mesh_frame])
            return combined_pcd

        # Create the point cloud
        # pcd_obs = o3d.geometry.PointCloud()
        # pcd_obs.points = o3d.utility.Vector3dVector(obs_p)
        # # o3d.visualization.draw_geometries([pcd_obs])
        # o3d.io.write_point_cloud("obs_pcd.ply", pcd_obs)

        # pcd_free = o3d.geometry.PointCloud()
        # pcd_free.points = o3d.utility.Vector3dVector(free_p)
        # # o3d.visualization.draw_geometries([pcd_free])
        # o3d.io.write_point_cloud("free_pcd.ply", pcd_free)

        # combine_colored_pcds(pcd_obs, pcd_free)

        self.UpdatePCDwithPs(obs_p, free_p, is_downsample=True)
        if self._cfg_tsdf.filter_outliers:
            obs_p = self.FilterCloud(self.obs_points, outlier_filter=False)
            free_p = self.FilterCloud(self.free_points, outlier_filter=False)
            self.UpdatePCDwithPs(obs_p, free_p)

        # combine_colored_pcds(self.obs_pcd, self.free_pcd)
        self.UpdateMapParams()
        return

    def TerrainAnalysisNew(self, obs_p):
        indices = np.where(obs_p[:, 2] < self._cfg_tsdf.robot_height * self._cfg_tsdf.robot_height_factor)[0]
        
        return obs_p[indices, :]
        # # naive approach with z values
        # for p in obs_p:
        #     p_height = p[2] #+ self._cfg_tsdf.offset_z
        #     if (p_height < self._cfg_tsdf.robot_height * self._cfg_tsdf.robot_height_factor): 
        #         # remove ground and ceiling
        #         obs_points[obs_idx, :] = p
        #         obs_idx = obs_idx + 1
        # return obs_points[:obs_idx, :]


    def TerrainAnalysis(self, input_points):
        obs_points = np.zeros(input_points.shape)
        free_points = np.zeros(input_points.shape)
        obs_idx = 0
        free_idx = 0
        # naive approach with z values
        for p in input_points:
            p_height = p[2] + self._cfg_tsdf.offset_z
            if (p_height > self._cfg_tsdf.ground_height * 1.0) and (
                p_height
                < self._cfg_tsdf.robot_height * self._cfg_tsdf.robot_height_factor
            ):  # remove ground and ceiling
                obs_points[obs_idx, :] = p
                obs_idx = obs_idx + 1
            elif p_height < self._cfg_tsdf.ground_height:
                free_points[free_idx, :] = p
                free_idx = free_idx + 1
        return obs_points[:obs_idx, :], free_points[:free_idx, :]

    def UpdateMapParams(self):
        # breakpoint()
        if self.obs_points.shape[0] == 0:
            # print("No points received.")
            return
        max_x_obs, max_y_obs, _ = (
            np.amax(self.obs_points, axis=0) + self._cfg_general.clear_dist
        )
        min_x_obs, min_y_obs, _ = (
            np.amin(self.obs_points, axis=0) - self._cfg_general.clear_dist
        )

        self.closest_possible_obs_point = np.array([[min_x_obs + self._cfg_general.clear_dist,
                                                     min_y_obs]])
        
        max_x_free, max_y_free, min_x_free, min_y_free = 0.0, 0.0, 0.0, 0.0
        if len(self.free_points) > 0:
            max_x_free, max_y_free, _ = (
                np.amax(self.free_points, axis=0)# + self._cfg_general.clear_dist
            )
            min_x_free, min_y_free, _ = (
                np.amin(self.free_points, axis=0)# - self._cfg_general.clear_dist
            )

        max_x = max(max_x_obs, max_x_free)
        max_y = max(max_y_obs, max_y_free)
        min_x = min(min(min_x_obs, min_x_free), 0.0)
        min_y = min(min_y_obs, min_y_free)

        self.num_x = (
            np.ceil((max_x - min_x) / self._cfg_general.resolution / 10).astype(int)
            * 10
        )
        self.num_y = (
            np.ceil((max_y - min_y) / self._cfg_general.resolution / 10).astype(int)
            * 10
        )
        self.start_x = (
            max_x + min_x
        ) / 2.0 - self.num_x / 2.0 * self._cfg_general.resolution
        self.start_y = (
            max_y + min_y
        ) / 2.0 - self.num_y / 2.0 * self._cfg_general.resolution

        # for small map
        self.num_x_small = (
            np.ceil((max_x_obs - min_x_obs) / self._cfg_general.resolution / 10).astype(int)
            * 10
        )
        self.num_y_small = (
            np.ceil((max_y_obs - min_y_obs) / self._cfg_general.resolution / 10).astype(int)
            * 10
        )
        self.start_x_small = (
            max_x_obs + min_x_obs
        ) / 2.0 - self.num_x_small / 2.0 * self._cfg_general.resolution
        self.start_y_small = (
            max_y_obs + min_y_obs
        ) / 2.0 - self.num_y_small / 2.0 * self._cfg_general.resolution

        # print("tsdf small map initialized, with size: %d, %d" % (self.num_x_small, self.num_y_small))
        # print("tsdf map initialized, with size: %d, %d" % (self.num_x, self.num_y))
        self.is_map_ready = True

    def zero_single_isolated_ones(self, arr):
        neighbors = ndimage.convolve(
            arr, weights=np.ones((3, 3)), mode="constant", cval=0
        )
        isolated_ones = (arr == 1) & (neighbors == 1)
        arr[isolated_ones] = 0
        return arr

    def CreateTSDFMap(self):
        # breakpoint()
        if not self.is_map_ready:
            raise ValueError("create tsdf map fails, no points received.")
        free_map = np.zeros([self.num_x, self.num_y])
        free_map_small = np.ones([self.num_x_small, self.num_y_small])
        obs_map_small = np.zeros([self.num_x_small, self.num_y_small])
        free_I = self.IndexArrayOfPs(self.free_points, self.start_x_small, self.start_y_small)
        obs_I = self.IndexArrayOfPs(self.obs_points, self.start_x_small, self.start_y_small)
        # create free place map
        for i in obs_I:
            obs_map_small[i[0], i[1]] = 1.0
        obs_map_small = self.zero_single_isolated_ones(obs_map_small)
        obs_map_small = gaussian_filter(obs_map_small, sigma=self._cfg_tsdf.sigma_expand)

        for i in free_I:
            if 0 < i[0] < self.num_x_small and 0 < i[1] < self.num_y_small:
                try:
                    free_map_small[i[0], i[1]] = 0
                except:
                    import ipdb

                    ipdb.set_trace()
        free_map_small = gaussian_filter(free_map_small, sigma=self._cfg_tsdf.sigma_expand)

        free_map_small[free_map_small < self._cfg_tsdf.free_space_threshold] = 0
        # assign obstacles
        free_map_small[obs_map_small > self._cfg_tsdf.obstacle_threshold] = 1.0
        # print("occupancy map generation completed.")

        # fill up free map with small map
        array_of_indices_small_map = np.array(
            [
                [i, j]
                for i in range(free_map_small.shape[0])
                for j in range(free_map_small.shape[1])
            ]
        )
        points_small_map = self.IndexToPos(
            array_of_indices_small_map, self.start_x_small, self.start_y_small
        )
        indices_in_big_map = self.IndexArrayOfPs(
            points_small_map, self.start_x, self.start_y
        )
        for idx, i in enumerate(indices_in_big_map):
            if 0 <= i[0] < self.num_x and 0 <= i[1] < self.num_y:
                free_map[i[0], i[1]] = free_map_small[
                    array_of_indices_small_map[idx, 0],
                    array_of_indices_small_map[idx, 1],
                ]
        # fill the indices less the start_x_small and start_y_small as free

        # bottom_left_corner_in_small_map_point = np.array(
        #     [self.start_x_small, self.start_y_small + self.num_y_small * self._cfg_general.resolution]
        # )
        # bottom_right_corner_req_point = np.array(
        #     [0.0, self.start_y_small]
        # )
        # upper_left_corner_req_indices = self.IndexArrayOfPs(
        #     np.array([bottom_left_corner_in_small_map_point]), self.start_x, self.start_y
        # )[0]
        # bottom_right_corner_req_indices = self.IndexArrayOfPs(
        #     np.array([bottom_right_corner_req_point]), self.start_x, self.start_y
        # )[0]

        # # make points between these two corners as free
        # # breakpoint()
        # free_map[0:upper_left_corner_req_indices[0], 
        #          bottom_right_corner_req_indices[1]:upper_left_corner_req_indices[1]] = 0.0
        
        # Zero all free points below closest possible obstacle point
        closest_obs_point_x_idx = self.IndexArrayOfPs(self.closest_possible_obs_point, 
                                                        self.start_x, self.start_y)[0]
        free_map[0:closest_obs_point_x_idx[0], :] = 0.0

        # Distance Transform
        tsdf_array_dist_to_free_space = ndimage.distance_transform_edt(free_map)
        tsdf_array_dist_to_free_space[tsdf_array_dist_to_free_space > 0.0] = \
        np.log(tsdf_array_dist_to_free_space[tsdf_array_dist_to_free_space > 0.0] + math.e)

        # flipped = np.flip(np.flip(tsdf_array_dist_to_free_space, axis=1), axis=0)
        # plt.imshow(flipped, cmap="viridis", vmin=0.0, vmax=5.0, aspect="auto")
        # plt.colorbar(label="Free space")
        # plt.title("Free Map")

        # plt.show()

        inverted_free_map = 1 - free_map

        tsdf_array_distance_to_obstacle = ndimage.distance_transform_edt(inverted_free_map)
        tsdf_array_distance_to_obstacle[tsdf_array_distance_to_obstacle > 0.0] = \
            np.log(tsdf_array_distance_to_obstacle[tsdf_array_distance_to_obstacle > 0.0] + math.e)

        # flipped = np.flip(np.flip(tsdf_array_distance_to_obstacle, axis=1), axis=0)
        # plt.imshow(flipped, cmap="viridis", vmin=0.0, vmax=5.0, aspect="auto")
        # plt.colorbar(label="Obs space")
        # plt.title("Obs Map")

        # plt.show()

        tsdf_array = np.where(tsdf_array_distance_to_obstacle == 0, 
                              -tsdf_array_dist_to_free_space, tsdf_array_distance_to_obstacle)

        # tsdf_array[tsdf_array > 0.0] = np.log(tsdf_array[tsdf_array > 0.0] + math.e)
        tsdf_array = gaussian_filter(tsdf_array, sigma=self._cfg_general.sigma_smooth)
        closest_obs_point_x_idx = self.IndexArrayOfPs(self.closest_possible_obs_point, 
                                                self.start_x, self.start_y)[0]
        # tsdf_array[0:closest_obs_point_x_idx[0], :] = tsdf_array[closest_obs_point_x_idx[0], :]

        # flipped = np.flip(np.flip(tsdf_array, axis=1), axis=0)
        # plt.imshow(flipped, cmap="viridis", aspect="auto")
        # plt.colorbar(label="TSDF Values")
        # plt.title("Final TSDF Map")

        # plt.show()

        viz_points = np.concatenate((self.obs_points, self.free_points), axis=0)

        ground_array = np.ones([self.num_x, self.num_y]) * 0.0

        return [tsdf_array, viz_points, ground_array], [
            float(self.start_x),
            float(self.start_y),
        ]

    def IndexArrayOfPs(self, points, start_x, start_y):
        indexes = points[:, :2] - np.array([start_x, start_y])
        indexes = (np.round(indexes / self._cfg_general.resolution)).astype(int)
        return indexes
    
    def IndexToPos(self, array_of_indices, start_x, start_y):
        positions = array_of_indices * self._cfg_general.resolution + np.array([start_x, start_y])
        return positions
    

    def FilterCloud(self, points, outlier_filter=True):
        # crop points
        if any(
            [
                self._cfg_general.x_max,
                self._cfg_general.x_min,
                self._cfg_general.y_max,
                self._cfg_general.y_min,
            ]
        ):
            points_x_idx_upper = (
                (points[:, 0] < self._cfg_general.x_max)
                if self._cfg_general.x_max is not None
                else np.ones(points.shape[0], dtype=bool)
            )
            points_x_idx_lower = (
                (points[:, 0] > self._cfg_general.x_min)
                if self._cfg_general.x_min is not None
                else np.ones(points.shape[0], dtype=bool)
            )
            points_y_idx_upper = (
                (points[:, 1] < self._cfg_general.y_max)
                if self._cfg_general.y_max is not None
                else np.ones(points.shape[0], dtype=bool)
            )
            points_y_idx_lower = (
                (points[:, 1] > self._cfg_general.y_min)
                if self._cfg_general.y_min is not None
                else np.ones(points.shape[0], dtype=bool)
            )
            points = points[
                np.vstack(
                    (
                        points_x_idx_lower,
                        points_x_idx_upper,
                        points_y_idx_upper,
                        points_y_idx_lower,
                    )
                ).all(axis=0)
            ]

        if outlier_filter:
            # Filter outlier in points
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            cl, _ = pcd.remove_statistical_outlier(
                nb_neighbors=self._cfg_tsdf.nb_neighbors,
                std_ratio=self._cfg_tsdf.std_ratio,
            )
            points = np.asarray(cl.points)

        return points

    def VizCloud(self, pcd):
        o3d.visualization.draw_geometries([pcd])  # visualize point cloud


# EoF