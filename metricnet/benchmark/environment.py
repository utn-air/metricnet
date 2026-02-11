import copy
import os
from typing import Any, Dict, List, Optional, Tuple

import click
import habitat_sim
import magnum as mn
import numpy as np
import scipy
import tqdm
import pickle
from PIL import Image
from scipy.spatial.transform import Rotation as R

from metricnet.benchmark.environment_utils import from_quat_to_angle, update_rotations


class HabitatEnvMultiAgent:
    def __init__(
        self,
        scene_id_path: str,
        image_width: int = 640,
        image_height: int = 480,
        camera_height: np.ndarray = np.array([0.5]),
        fisheye_xi: np.ndarray = np.array([0.0]),
        fisheye_alpha: np.ndarray = np.array([0.0]),
        focal_length: np.ndarray = np.array([364.8]),
        agent_ids: List[int] = [0],
        fps: float = 60,
        context_size: int = 3,
        max_distance_between_images_in_topomap: float = 0.25,
        min_topomap_distance: float = 7.5,
    ) -> None:
        # define some common simulation settings components
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.enable_physics = True
        assert os.path.exists(scene_id_path), f"File {scene_id_path} does not exist"
        sim_cfg.scene_id = scene_id_path

        # define agent configuration
        agent_configs = []
        for i in range(len(agent_ids)):
            fisheye_sensor_spec = habitat_sim.FisheyeSensorDoubleSphereSpec()
            fisheye_sensor_spec.uuid = "fisheye_sensor"
            fisheye_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            fisheye_sensor_spec.sensor_model_type = (
                habitat_sim.FisheyeSensorModelType.DOUBLE_SPHERE
            )
            fisheye_sensor_spec.xi = fisheye_xi[i]
            fisheye_sensor_spec.alpha = fisheye_alpha[i]
            fisheye_sensor_spec.focal_length = [focal_length[i], focal_length[i]]
            fisheye_sensor_spec.resolution = mn.Vector2i([image_height, image_width])
            fisheye_sensor_spec.position = mn.Vector3(0, camera_height[i], 0)
            fisheye_sensor_spec.principal_point_offset = [
                i / 2 for i in fisheye_sensor_spec.resolution
            ]
            agent_cfg = habitat_sim.agent.AgentConfiguration()
            agent_cfg.sensor_specifications = [fisheye_sensor_spec]
            agent_configs.append(agent_cfg)
        cfg = habitat_sim.Configuration(sim_cfg, agent_configs)

        # define simulator configuration
        self.fps = fps

        # initialize the agent
        self.sims = habitat_sim.Simulator(cfg)
        self.agents = [self.sims.initialize_agent(id) for id in agent_ids]
        self.agent_seeds = [-1 for id in agent_ids]
        self.velocity_controls = [
            habitat_sim.physics.VelocityControl() for _ in agent_ids
        ]
        self.sims.config.sim_cfg.allow_sliding = False
        for vel_ctrl in self.velocity_controls:
            vel_ctrl.controlling_lin_vel = True
            vel_ctrl.lin_vel_is_local = True
            vel_ctrl.controlling_ang_vel = True
            vel_ctrl.ang_vel_is_local = True
        self.agent_ids = agent_ids
        self.max_distance_between_images_in_topomap = (
            max_distance_between_images_in_topomap
        )
        self.min_topomap_distance = min_topomap_distance

        # context queue
        self.context_size = context_size
        self.context_queue = [[] for _ in agent_ids]
        self.topomap = [[] for _ in agent_ids]
        self.topomap_positons = [[] for _ in agent_ids]
        self.topomap_agent_states = [[] for _ in agent_ids]
        self.trajectories = [[] for _ in agent_ids]
        self.navigable_points = []
        for _ in range(1000):
            p = self.sims.pathfinder.get_random_navigable_point()
            self.navigable_points.append(np.array([p[0], p[1], p[2]]))
        self.closest_nodes = [0 for _ in self.agent_ids]
        self.goal_nodes = [-1 for _ in self.agent_ids]

        self.cur_pose = [None for _ in self.agent_ids]
        self.goal_pose = [None for _ in self.agent_ids]
        self.reset()

    def save_pose(
        self,
        agent_id: int,
        agent_state: habitat_sim.AgentState,
        save_type: str = "current",
    ) -> None:
        pos = agent_state.position
        orn = agent_state.rotation
        w, x, y, z = orn.w, orn.x, orn.y, orn.z
        euler_orn = R.from_quat([x, y, z, w]).as_euler("zxy", degrees=True)
        if save_type == "current":
            self.cur_pose[agent_id] = np.hstack((pos, euler_orn))
            self.trajectories[agent_id].append(np.hstack((pos, euler_orn)))
        elif save_type == "goal":
            self.goal_pose[agent_id] = np.hstack((pos, euler_orn))

    def get_context_queue(self, agent_id: int) -> List[Image.Image]:
        obs = self.get_current_observation(agent_id)
        image = Image.fromarray(obs)
        if len(self.context_queue[agent_id]) == 0:
            self.context_queue[agent_id] = [image for i in range(self.context_size + 1)]
        else:
            self.context_queue[agent_id].pop(0)
            self.context_queue[agent_id].append(image)
        return self.context_queue[agent_id]

    def get_multi_agent_context_queue(
        self, agent_ids: Optional[List[int]] = None
    ) -> List[Image.Image]:
        if agent_ids is None:
            agent_ids = self.agent_ids
        multi_agent_context_queue = []
        for agent_id in agent_ids:
            multi_agent_context_queue += self.get_context_queue(agent_id)
        return multi_agent_context_queue

    def clear_context_queue(self, agent_id: int) -> None:
        self.context_queue[agent_id] = []

    def reset_agent(self, agent_id: int) -> None:
        # set seed for random navigable point
        self.sims.reset(agent_id)
        agent_seed = np.random.randint(1, 1000)
        self.agent_seeds[agent_id] = agent_seed
        self.sims.pathfinder.seed(agent_seed)

        # spawn the agent
        agent_state = habitat_sim.AgentState()

        # spawn the agent at a random navigable point
        agent_state.position = self.sims.pathfinder.get_random_navigable_point()
        agent_state.rotation = habitat_sim.utils.common.quat_from_magnum(
            mn.Quaternion.rotation(mn.Rad(-np.pi), mn.Vector3(0, 1, 0))
        )
        self.agents[agent_id].set_state(agent_state)

        # empty the context queue and trajectory
        self.context_queue[agent_id] = []
        self.trajectories[agent_id] = []

    def reset(self) -> None:
        # setup the navmesh
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.edge_max_error = 1.3
        navmesh_settings.agent_radius = 0.3
        navmesh_settings.edge_max_len = 0.3 * 8
        navmesh_success = self.sims.recompute_navmesh(
            self.sims.pathfinder, navmesh_settings
        )
        assert navmesh_success, "Error: could not build navmesh"
        for id in self.agent_ids:
            self.reset_agent(id)
        for idx in range(len(self.context_queue)):
            self.context_queue[idx] = []
            self.trajectories[idx] = []

    def check_collision(
        self,
        end_pos: np.ndarray,
        prev_state: habitat_sim.RigidState,
        next_state: habitat_sim.RigidState,
    ) -> bool:
        # check if a collision occurred by monitoring distances moved by agent
        dist_moved_before_filter = (
            next_state.translation - prev_state.translation
        ).dot()
        dist_moved_after_filter = (end_pos - prev_state.translation).dot()
        collided = (dist_moved_after_filter + 1e-5) < dist_moved_before_filter
        return collided

    def get_current_observation(self, agent_id: int) -> np.ndarray:
        cur_obs = self.sims.get_sensor_observations(agent_id)
        return cur_obs["fisheye_sensor"][:, :, :3]

    def step(self, action: np.ndarray, controller: str = "velocity") -> List[bool]:
        if controller == "teleport":  # position control / flash controller
            teleport_pos, teleport_rot = self.convert_ego_to_world(action)
            for action_id in range(action.shape[1]):
                collided = self.teleport_control(
                    teleport_point=teleport_pos[:, action_id],
                    teleport_rotation=teleport_rot[:, action_id],
                )
        elif controller == "velocity":
            x, y = action[:, -1, 0], action[:, -1, 1]
            dist = np.linalg.norm([x, y], axis=0)
            theta = np.arctan2(y, x)
            v = dist
            w = theta
            for agent_id in self.agent_ids:
                self.velocity_controls[agent_id].linear_velocity = np.array(
                    [0.0, 0.0, -v[agent_id]]
                )
                self.velocity_controls[agent_id].angular_velocity = np.array(
                    [0.0, w[agent_id], 0.0]
                )
            collided = self.velocity_control(
                max_steps=6
            )  # only simulate for 6 steps, matches NoMaD recompute frequency
        return collided

    def convert_ego_to_world(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # collect points in robot frame
        point_in_robot_frame = np.stack(
            [
                -actions[:, :, 1],
                np.zeros((actions.shape[0], actions.shape[1])),
                -actions[:, :, 0],
                np.ones((actions.shape[0], actions.shape[1])),
            ],
            axis=1,
        )
        # get the transformation matrix for each agent (4x4
        transformations = np.empty((actions.shape[0], 4, 4), dtype=np.float32)
        rotations = np.empty((actions.shape[0]), dtype=np.float32)
        translations = np.empty((actions.shape[0], 3), dtype=np.float32)
        for i in range(actions.shape[0]):
            rotations[i] = from_quat_to_angle(self.agents[i].state.rotation)
            translations[i] = self.agents[i].state.position

        # create the transformation matrix for each agent
        transformations = np.stack(
            [
                np.stack(
                    [
                        np.cos(rotations),
                        np.zeros_like(rotations),
                        np.sin(rotations),
                        translations[:, 0],
                    ],
                    axis=-1,
                ),
                np.stack(
                    [
                        np.zeros_like(rotations),
                        np.ones_like(rotations),
                        np.zeros_like(rotations),
                        translations[:, 1],
                    ],
                    axis=-1,
                ),
                np.stack(
                    [
                        -np.sin(rotations),
                        np.zeros_like(rotations),
                        np.cos(rotations),
                        translations[:, 2],
                    ],
                    axis=-1,
                ),
                np.array([[0, 0, 0, 1]] * actions.shape[0], dtype=np.float32),
            ],
            axis=1,
        )

        # apply the transformation to the points in robot frame
        point_in_world_frame = np.moveaxis(
            np.matmul(transformations, point_in_robot_frame), 1, 2
        )[:, :, :3]
        point_in_world_frame = np.concatenate(
            [translations[:, None, :], point_in_world_frame], axis=1
        )

        # compute the new rotations
        dx = -point_in_world_frame[:, 1:, 0] + point_in_world_frame[:, :-1, 0]
        dz = -point_in_world_frame[:, 1:, 2] + point_in_world_frame[:, :-1, 2]
        stationary = np.logical_and(np.isclose(dx, 0), np.isclose(dz, 0))
        new_rotations = np.arctan2(
            -point_in_world_frame[:, 1:, 0] + point_in_world_frame[:, :-1, 0],
            -point_in_world_frame[:, 1:, 2] + point_in_world_frame[:, :-1, 2],
        )

        # update the rotations
        if stationary.any():
            click.echo(click.style("Warning: some agents are stationary", fg="yellow"))
            new_rotations = update_rotations(stationary, new_rotations, rotations)

        return point_in_world_frame[:, 1:, :], new_rotations

    def velocity_control(self, max_steps: int = 20) -> List[bool]:
        collision_info = [False for _ in self.agent_ids]
        for step_idx in range(self.fps):
            for agent_id in self.agent_ids:
                agent_state = self.agents[agent_id].state
                # move the agent by using the velocity control attribute
                prev_state = habitat_sim.RigidState(
                    habitat_sim.utils.common.quat_to_magnum(agent_state.rotation),
                    agent_state.position,
                )
                next_state = self.velocity_controls[agent_id].integrate_transform(
                    self.dt, prev_state
                )
                # apply the next state to the agent and check for the closest navigation point
                end_pos = self.sims.step_filter(
                    prev_state.translation, next_state.translation
                )
                agent_state.position = end_pos
                agent_state.rotation = habitat_sim.utils.common.quat_from_magnum(
                    next_state.rotation
                )
                self.agents[agent_id].set_state(agent_state)
                self.save_pose(
                    agent_id=agent_id, agent_state=agent_state, save_type="current"
                )
                # check for collisions
                collided = self.check_collision(
                    prev_state=prev_state, next_state=next_state, end_pos=end_pos
                )
                if collided:
                    collision_info[agent_id] = True

            if step_idx >= max_steps:  # only simulate for max_steps
                return collision_info

        return collision_info

    def teleport_control(
        self, teleport_point: np.ndarray, teleport_rotation: np.ndarray
    ) -> List[bool]:
        collision_info = [False for _ in self.agent_ids]
        for agent_id in self.agent_ids:
            # teleport the agent to the target point and rotation
            agent_state = self.agents[agent_id].state
            prev_state = habitat_sim.RigidState(
                habitat_sim.utils.common.quat_to_magnum(agent_state.rotation),
                agent_state.position,
            )
            next_state = habitat_sim.RigidState(
                habitat_sim.utils.common.quat_to_magnum(
                    habitat_sim.utils.common.quat_from_magnum(
                        mn.Quaternion.rotation(
                            mn.Rad(teleport_rotation[agent_id]),
                            mn.Vector3(0, 1, 0),
                        )
                    )
                ),
                mn.Vector3(
                    teleport_point[agent_id, 0],
                    agent_state.position[1],
                    teleport_point[agent_id, 2],
                ),
            )

            # apply the next state to the agent and check for the closest navigation point
            end_pos = self.sims.step_filter(
                prev_state.translation, next_state.translation
            )
            agent_state.position = end_pos
            agent_state.rotation = habitat_sim.utils.common.quat_from_magnum(
                next_state.rotation
            )
            self.agents[agent_id].set_state(agent_state)
            self.save_pose(
                agent_id=agent_id, agent_state=agent_state, save_type="current"
            )

            # check for collisions
            collided = self.check_collision(
                prev_state=prev_state, next_state=next_state, end_pos=end_pos
            )
            if collided:
                collision_info[agent_id] = True

        return collision_info

    def get_path(self) -> habitat_sim.ShortestPath:
        # initialize shortest path object
        path = habitat_sim.ShortestPath()
        found_path = False

        while not found_path:
            # randomly select start and end points that are at least min_total_length apart
            start_idx = np.random.randint(0, len(self.navigable_points))
            start_pos = self.navigable_points[start_idx]
            path.requested_start = start_pos

            # compute distance from start to all other points
            diff = start_pos - self.navigable_points
            dist = np.linalg.norm(diff, axis=1)

            # select all points that are at least min_total_length away
            end_indices = np.where(dist > self.min_topomap_distance)[0]
            if len(end_indices) == 0:
                continue
            end_idx = np.random.choice(end_indices)
            path.requested_end = self.navigable_points[end_idx]
            found_path = self.sims.pathfinder.find_path(path)

            # if Z-diff in path is too large, try again
            if found_path:
                # compute Z difference between all points in the path.points array
                path_points_height = np.array(path.points)[:, 1]
                z_diff = abs(min(path_points_height) - max(path_points_height))
                if z_diff > 0.05:
                    found_path = False
        return path

    def update_agent_states(self, pos: np.ndarray, quat: np.ndarray) -> None:
        for agent_id in self.agent_ids:
            agent_state = habitat_sim.AgentState()
            agent_state.rotation = habitat_sim.utils.common.quat_from_coeffs(
                quat[agent_id]
            )
            agent_state.position = pos[agent_id]
            self.agents[agent_id].set_state(agent_state)

    def collect_topomap(
        self,
        agent_id: int,
        edge_max_len: float = 0.3 * 8,
        edge_max_error: float = 1.3,
        agent_radius: float = 0.3,
        save_path: Optional[str] = None,
    ) -> List[habitat_sim.AgentState]:
        # check if save_path is a folder that exists
        os.makedirs(save_path, exist_ok=True)

        # create a new navmesh settings
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.edge_max_len = edge_max_len
        navmesh_settings.edge_max_error = edge_max_error
        navmesh_settings.agent_radius = agent_radius
        navmesh_success = self.sims.recompute_navmesh(
            self.sims.pathfinder, navmesh_settings
        )
        assert navmesh_success, "Error: could not build navmesh"

        # get the points along the path
        path = self.get_path(agent_id=agent_id)
        path_points = path.points
        if len(path_points) < 2:
            raise ValueError(
                f"Path points are too few: {len(path_points)} < 2, "
                f"try increasing the min_topomap_distance or max_distance_between_images_in_topomap"
            )

        # interpolate path points until we have min_topo_size points
        path_points = np.array(path_points)
        distances = np.cumsum(np.linalg.norm(np.diff(path_points, axis=0), axis=1))
        distances = np.insert(distances, 0, 0)  # add the starting point
        path_points = scipy.interpolate.interp1d(distances, path_points, axis=0)(
            np.arange(
                0,
                distances[-1],
                self.max_distance_between_images_in_topomap,
            )
        )

        # init the topomap
        agent_state = habitat_sim.AgentState()
        topomap = []
        self.topomap_agent_states[agent_id] = []
        self.topomap[agent_id] = []
        self.topomap_positons[agent_id] = []
        self.trajectories[agent_id] = []

        # traverse the path and collect images
        for i in range(0, len(path_points)):
            agent_state.position = path_points[i]
            if i > 0:
                angle = np.arctan2(
                    -path_points[i][0] + path_points[i - 1][0],
                    -path_points[i][2] + path_points[i - 1][2],
                )
            if i == 0:
                angle = np.arctan2(
                    -path_points[i + 1][0] + path_points[i][0],
                    -path_points[i + 1][2] + path_points[i][2],
                )
            rotation = habitat_sim.utils.common.quat_from_magnum(
                mn.Quaternion.rotation(mn.Rad(angle), mn.Vector3(0, 1, 0))
            )
            agent_state.rotation = rotation
            self.agents[agent_id].set_state(agent_state)
            self.topomap_agent_states[agent_id].append(copy.deepcopy(agent_state))
            obs = Image.fromarray(self.get_current_observation(agent_id=agent_id))
            topomap.append(obs)
            obs.save(f"{save_path}/{i}.png")

        # set topomap in the agent
        self.topomap_positons[agent_id] = path_points
        self.topomap[agent_id] = topomap
        return self.topomap_agent_states

    def close_all(self) -> None:
        self.sims.close()

    def save_dataset_info(self, agent_id: int) -> Dict[str, Any]:
        params = {
            "image_width": self.sims._sensors["fisheye_sensor"]._spec.resolution[1],
            "image_height": self.sims._sensors["fisheye_sensor"]._spec.resolution[0],
            "camera_height": self.sims.agents[agent_id]
            .agent_config.sensor_specifications[0]
            .position[1],
            "fisheye_xi": self.sims.agents[agent_id]
            .agent_config.sensor_specifications[0]
            .xi,
            "fisheye_alpha": self.sims.agents[agent_id]
            .agent_config.sensor_specifications[0]
            .alpha,
            "focal_length": self.sims.agents[agent_id]
            .agent_config.sensor_specifications[0]
            .focal_length[0],
            "max_distance_between_images_in_topomap": self.max_distance_between_images_in_topomap,
        }
        return params

    def save_topomap(self, save_path, agent_id):
        os.makedirs(save_path, exist_ok=True)
        for idx, im in enumerate(
            tqdm.tqdm(
                self.topomap[agent_id],
                total=len(self.topomap[agent_id]),
            )
        ):
            im.convert("RGB").save(f"{save_path}/{idx}.jpg")
        poses = self.convert_topomap_pose_to_array(agent_id)
        traj_data = {
            "position": -poses[:, [2, 0]],
            "yaw": poses[:, 3],
        }
        with open(os.path.join(save_path, "traj_data.pkl"), "wb") as f:
            pickle.dump(traj_data, f)

    def convert_topomap_pose_to_array(self, agent_id):
        poses = np.empty(
            (len(self.topomap_agent_states[agent_id]), 4), dtype=np.float32
        )
        for idx, agent_state in enumerate(self.topomap_agent_states[agent_id]):
            position = agent_state.position
            rotation = agent_state.rotation
            w, x, y, z = rotation.w, rotation.x, rotation.y, rotation.z
            euler_orn = R.from_quat([x, y, z, w]).as_euler("zxy", degrees=False)
            poses[idx, 0] = position[0]
            poses[idx, 1] = position[1]
            poses[idx, 2] = position[2]
            poses[idx, 3] = euler_orn[2]  # yaw angle
        return poses

    def get_agents_positions(self):
        positions = []
        for agent_id in self.agent_ids:
            agent_state = self.agents[agent_id].state
            positions.append(agent_state.position)
        return np.array(positions)

    def get_max_topomap_to_trajectory(self, agent_id: int, topomap: np.ndarray) -> int:
        if len(self.trajectories[agent_id]) == 0:
            return 0
        topomap_points = topomap[:, [0, 2]]
        trajectory_points = np.array(self.trajectories[agent_id])[:, [0, 2]]

        diff = topomap_points[:, None, :] - trajectory_points[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)

        idx = 0
        for i in range(topomap_points.shape[0]):
            if dists[i].min() < 0.5:
                idx = i
        return idx
