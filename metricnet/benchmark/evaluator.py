import json
import multiprocessing
import os
import re
from pathlib import Path
from typing import List, Optional

import click
import numpy as np
import torch
import tqdm
from PIL import Image

from metricnet.benchmark.agent import NavAgent
from metricnet.benchmark.environment import HabitatEnvMultiAgent
from metricnet.benchmark.evaluator_utils import (
    get_action_prediction,
    get_scaled_trajectory,
)
from metricnet.benchmark.models_utils import (
    load_config,
    transform_images_nomad,
    transform_obs_metricnet,
)
from metricnet.models.metricnav.guide import PathGuide as MetricNavPathGuide
from metricnet.models.navidiffusor.guide import PathGuide
from metricnet.training.utils import ACTION_STATS


class HabitatEvaluator:
    def __init__(
        self,
        benchmark_dir: str,
        model_type: str,
        model_cfg: str,
        model_ckpt: str,
        result_dir: str,
        repetitions_per_trajectory: int = 3,
        num_workers: int = 5,
        max_n_agents: int = 3,
        eval_type: str = "nav",
        device: str = "cuda:0",
        controller: str = "velocity",
        waypoint_idx: int = 3,
        scale: float = 0.25,
    ) -> None:
        # load all scenes
        benchmark_dir = os.path.expandvars(benchmark_dir)
        self.scenes = sorted(
            [
                x
                for x in os.listdir(benchmark_dir)
                if os.path.isdir(os.path.join(benchmark_dir, x))
                and "scene_config.yaml" in os.listdir(os.path.join(benchmark_dir, x))
            ]
        )
        click.echo(click.style(f">> Found {len(self.scenes)} scenes", fg="green"))

        # set variables
        self.eval_type = eval_type
        self.model_cfg = load_config(model_cfg)
        self.model_type = model_type
        self.model_ckpt = model_ckpt
        self.device = device
        self.eval_type = eval_type
        self.repetitions_per_trajectory = repetitions_per_trajectory
        self.waypoint_idx = waypoint_idx
        self.result_dir = result_dir
        self.num_workers = num_workers
        self.max_n_agents = max_n_agents
        self.controller = controller
        self.scale = scale

        # set seeds
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        np.random.seed(0)
        self.seeds = np.random.randint(0, 1000, size=(self.repetitions_per_trajectory))

        # run multiple scene parallely
        self.run_parallel_scene_evaluate(
            benchmark_dir, self.result_dir, model_cfg, model_ckpt, device=device
        )

    def run_parallel_scene_evaluate(
        self,
        benchmark_dir: str,
        result_dir: str,
        model_cfg: str,
        model_ckpt: str,
        device: str,
    ) -> None:
        args = [
            (
                f"{benchmark_dir}/{scene_name}",
                scene_name,
                result_dir,
                model_cfg,
                model_ckpt,
                self.max_n_agents,
                device,
                self.eval_type,
                self.controller,
                self.scale,
            )
            for scene_name in self.scenes
        ]
        if self.num_workers == 0:
            for scene_args in tqdm.tqdm(
                args, desc="Processing scenes", colour="magenta"
            ):
                self.run_model_on_scene(*scene_args)
        else:
            with multiprocessing.Pool(
                processes=min(self.num_workers, multiprocessing.cpu_count())
            ) as scene_pool:
                scene_pool.starmap(self.run_model_on_scene, args)

    def run_exploration(
        self,
        env: HabitatEnvMultiAgent,
        save_dir: str,
        model_cfg: str,
        model_ckpt: str,
        device: str = "cuda:0",
        controller: str = "velocity",
        batch_first_idx: int = 0,
        gt_scale: float = 0.25,
        repetition: int = 0,
    ) -> None:
        # set variables
        nav_agent = NavAgent(model_cfg, model_ckpt, device=device)
        num_agents = len(env.agent_ids)
        valid_until = [0 for _ in range(num_agents)]
        last_success = [True for _ in range(num_agents)]
        frames = [[] for _ in range(num_agents)]
        distance_explored = [0 for _ in range(num_agents)]
        agents_positions = env.get_agents_positions()
        k_steps = 10  # handcrafted for 10 steps (template for NoMaD and FlowNav)
        step_thresh = 500  # max steps per episode

        # set seeds for reproducibility
        torch.manual_seed(self.seeds[repetition])
        np.random.seed(self.seeds[repetition])

        # check if all agents are already done to skip (in case of re-run)
        all_agents_done = [False for _ in range(num_agents)]
        for agent_id in range(num_agents):
            all_agents_done[agent_id] = os.path.exists(
                f"{save_dir}/{(agent_id + batch_first_idx):04d}/{repetition}_{self.model_type}.json"
            )
        if all_agents_done == [True for _ in range(num_agents)]:
            click.echo(
                click.style(
                    f"All agents already done for repetition {repetition}, skipping...",
                    fg="yellow",
                )
            )
            return

        # create path guides
        if "navidiffusor" in self.model_type:
            pathguide = [
                PathGuide(device, ACTION_STATS, self.model_cfg["guidance_path_weights"])
                for _ in range(num_agents)
            ]
        elif "metricnav" in self.model_type:
            pathguide = [
                MetricNavPathGuide(
                    device, ACTION_STATS, self.model_cfg["guidance_path_weights"]
                )
                for _ in range(num_agents)
            ]
        else:
            pathguide = None

        for step_idx in tqdm.tqdm(range(step_thresh)):
            # check if all agents have failed to stop early
            if not np.any(last_success):
                click.echo(
                    click.style(
                        f"All agents failed at step {step_idx}. Stopping further steps for this trajectory.",
                        fg="red",
                    )
                )
                break

            # get observations
            context_queue = env.get_multi_agent_context_queue()
            obs_images = transform_images_nomad(
                context_queue, self.model_cfg["image_size"], center_crop=False
            )
            obs_images = torch.split(
                obs_images, 3 * (self.model_cfg["context_size"] + 1), dim=1
            )
            obs_images = torch.cat(obs_images, dim=0)
            obs_images = obs_images.to(device)
            mask = torch.ones(num_agents).long().to(device)
            repeat_lengths = [1 for _ in range(num_agents)]
            obs_im_rep = []
            mask_rep = []
            obs_now_pil = []
            for id in range(num_agents):
                obs_im_rep.append(
                    obs_images[id, :, :, :].repeat(repeat_lengths[id], 1, 1, 1)
                )
                mask_rep.append(mask[id].repeat(repeat_lengths[id]))
                obs_now_img = env.get_current_observation(id)
                obs_now_pil.append(Image.fromarray(obs_now_img))
            obs_now = transform_obs_metricnet(
                obs_now_pil, [224, 224], center_crop=False
            )
            obs_images = torch.cat(obs_im_rep, dim=0)
            mask = torch.cat(mask_rep, dim=0)

            # get conditioning
            obsgoal_cond = nav_agent.model(
                "vision_encoder",
                obs_img=obs_images,
                goal_img=torch.zeros(
                    (
                        num_agents,
                        3,
                        obs_images.shape[2],
                        obs_images.shape[3],
                    ),
                    device=device,
                ),
                input_goal_mask=mask,
            )
            obs_cond = []
            with torch.no_grad():
                for id in range(num_agents):
                    obs_cond.append(obsgoal_cond[id].unsqueeze(0))
                obs_cond = torch.cat(obs_cond, dim=0)
                noisy_action = torch.randn(
                    (num_agents, self.model_cfg["len_traj_pred"], 2), device=device
                )

            # predict action
            result = get_action_prediction(
                obs_cond=obs_cond,
                model_type=self.model_type,
                nav_agent=nav_agent,
                k_steps=k_steps,
                context_queue=context_queue,
                num_agents=num_agents,
                obs_now=obs_now,
                noisy_action=noisy_action,
                pathguide=pathguide,
                model_cfg=self.model_cfg,
                device=device,
            )

            # get final action
            naction = get_scaled_trajectory(
                model_type=self.model_type,
                naction=result["n_action"],
                nav_agent=nav_agent,
                num_agents=num_agents,
                model_cfg=self.model_cfg,
                obs_now=obs_now,
                pathguide=pathguide,
                device=device,
                gt_scale=gt_scale,
                goal_positions=result.get("goal_positions", None),
                best_goal_idx=result.get("best_goal_idx", None),
            )

            # step env
            collided = env.step(
                naction[:, : self.waypoint_idx + 1, :], controller=controller
            )
            new_agents_positions = env.get_agents_positions()

            # collect metrics
            for agent_id in range(num_agents):
                frames[agent_id].append(obs_now_pil[agent_id])
                if last_success[agent_id]:
                    distance_explored[agent_id] += np.linalg.norm(
                        new_agents_positions[agent_id] - agents_positions[agent_id]
                    )
                agents_positions[agent_id] = new_agents_positions[agent_id]

            # update success
            for id in range(num_agents):
                if not collided[id] and last_success[id]:
                    valid_until[id] = step_idx
                else:
                    last_success[id] = False

        self.log_info(
            save_dir=save_dir,
            frames=frames,
            reached_node=None,
            valid_until=valid_until,
            topomap_lengths=None,
            distance_explored=distance_explored,
            batch_first_idx=batch_first_idx,
            repetition=repetition,
            num_agents=num_agents,
            eval_type="exp",
        )
        env.close_all()

    def run_nav(
        self,
        env: HabitatEnvMultiAgent,
        agent_positions: List[np.ndarray],
        topomap: List[List[Image.Image]],
        save_dir: str,
        model_cfg: str,
        model_ckpt: str,
        device: str = "cuda:0",
        controller: str = "velocity",
        gt_scale: float = 0.25,
        batch_first_idx: int = 0,
        repetition: int = 0,
    ) -> None:
        # set variables
        nav_agent = NavAgent(model_cfg, model_ckpt, device=device)
        num_agents = len(env.agent_ids)
        topomap_lengths = [len(topomap[id]) for id in range(num_agents)]
        reached_node = [0 for _ in range(num_agents)]
        valid_until = [0 for _ in range(num_agents)]
        last_success = [True for _ in range(num_agents)]
        k_steps = 10
        step_thresh = 500
        goal_node = [self.waypoint_idx for _ in range(num_agents)]
        frames = [[] for _ in range(num_agents)]

        # set seeds for reproducibility
        torch.manual_seed(self.seeds[repetition])
        np.random.seed(self.seeds[repetition])

        # check if all agents are already done to skip (in case of re-run)
        all_agents_done = [False for _ in range(num_agents)]
        for agent_id in range(num_agents):
            all_agents_done[agent_id] = os.path.exists(
                f"{save_dir}/{(agent_id + batch_first_idx):04d}/{repetition}_{self.model_type}.json"
            )
        if all_agents_done == [True for _ in range(num_agents)]:
            click.echo(
                click.style(
                    f"All agents already done for repetition {repetition}, skipping...",
                    fg="yellow",
                )
            )
            return

        # get path guides
        if "navidiffusor" in self.model_type:
            pathguide = [PathGuide(device, ACTION_STATS) for _ in range(num_agents)]
        elif "metricnav" in self.model_type:
            pathguide = [
                MetricNavPathGuide(device, ACTION_STATS) for _ in range(num_agents)
            ]
        else:
            pathguide = None

        for step_idx in tqdm.tqdm(range(step_thresh)):
            # check if all agents have failed to stop early
            if not np.any(last_success):
                click.echo(
                    click.style(
                        f"All agents failed at step {step_idx}. Stopping further steps for this trajectory.",
                        fg="red",
                    )
                )
                break

            # get observations
            context_queue = env.get_multi_agent_context_queue()
            obs_images = transform_images_nomad(
                context_queue, self.model_cfg["image_size"], center_crop=False
            )
            obs_images = torch.split(
                obs_images, 3 * (self.model_cfg["context_size"] + 1), dim=1
            )
            obs_images = torch.cat(obs_images, dim=0)
            obs_images = obs_images.to(device)

            # update reached and goal nodes
            for i in range(num_agents):
                if last_success[i]:
                    agent_state = env.sims.get_agent(i).get_state()
                    pos = agent_state.position
                    distances = np.linalg.norm(
                        agent_positions[i][:, [0, 2]] - pos[[0, 2]], axis=1
                    )
                    closest_idx = np.argmin(distances)
                    if distances[closest_idx] <= 0.5 and closest_idx >= reached_node[i]:
                        reached_node[i] = env.get_max_topomap_to_trajectory(
                            i, agent_positions[i]
                        )

                    if closest_idx >= goal_node[i]:
                        goal_node[i] = min(
                            closest_idx + self.waypoint_idx, len(topomap[i]) - 1
                        )

            # prepare goal images
            goal_images = [[topomap[id][goal_node[id]]] for id in range(num_agents)]
            goal_images = [
                transform_images_nomad(
                    goal, self.model_cfg["image_size"], center_crop=False
                )
                for agent_goals in goal_images
                for goal in agent_goals
            ]
            goal_images = torch.concat(goal_images, dim=0)
            goal_images = goal_images.to(device)
            mask = torch.zeros(num_agents).long().to(device)
            repeat_lengths = [1 for id in range(num_agents)]
            obs_im_rep = []
            mask_rep = []
            obs_now_pil = []
            for id in range(num_agents):
                obs_im_rep.append(
                    obs_images[id, :, :, :].repeat(repeat_lengths[id], 1, 1, 1)
                )
                mask_rep.append(mask[id].repeat(repeat_lengths[id]))
                obs_now_img = env.get_current_observation(id)
                obs_now_pil.append(Image.fromarray(obs_now_img))
            obs_images = torch.cat(obs_im_rep, dim=0)
            mask = torch.cat(mask_rep, dim=0)
            obs_now = transform_obs_metricnet(
                obs_now_pil, [224, 224], center_crop=False
            )

            # get context conditioning
            obsgoal_cond = nav_agent.model(
                "vision_encoder",
                obs_img=obs_images,
                goal_img=goal_images,
                input_goal_mask=mask,
            )
            obs_cond = []
            with torch.no_grad():
                for id in range(num_agents):
                    obs_cond.append(obsgoal_cond[id].unsqueeze(0))
                obs_cond = torch.cat(obs_cond, dim=0)
                noisy_action = torch.randn(
                    (num_agents, self.model_cfg["len_traj_pred"], 2), device=device
                )

            # predict action
            result = get_action_prediction(
                obs_cond=obs_cond,
                model_type=self.model_type,
                nav_agent=nav_agent,
                k_steps=k_steps,
                context_queue=context_queue,
                num_agents=num_agents,
                obs_now=obs_now,
                noisy_action=noisy_action,
                pathguide=pathguide,
                model_cfg=self.model_cfg,
                device=device,
            )

            # get final action
            naction = get_scaled_trajectory(
                model_type=self.model_type,
                naction=result["n_action"],
                nav_agent=nav_agent,
                num_agents=num_agents,
                model_cfg=self.model_cfg,
                obs_now=obs_now,
                pathguide=pathguide,
                device=device,
                gt_scale=gt_scale,
                goal_positions=result.get("goal_positions", None),
                best_goal_idx=result.get("best_goal_idx", None),
            )

            # step env
            collided = env.step(
                naction[:, : self.waypoint_idx + 1, :], controller=controller
            )

            # add to video with goal
            for agent_id in range(num_agents):
                zero_pad = np.zeros((obs_now_pil[agent_id].size[1], 20, 3)).astype(
                    np.uint8
                )
                g = topomap[agent_id][goal_node[agent_id]]
                fin_im = np.hstack((obs_now_pil[agent_id], zero_pad, g))
                im = Image.fromarray(fin_im)
                frames[agent_id].append(im)

            # update success
            for id in range(num_agents):
                if not collided[id] and last_success[id]:
                    valid_until[id] = step_idx
                else:
                    if last_success[id]:
                        reached_node[id] = env.get_max_topomap_to_trajectory(
                            id, agent_positions[id]
                        )
                    last_success[id] = False
        self.log_info(
            save_dir=save_dir,
            frames=frames,
            reached_node=reached_node,
            valid_until=valid_until,
            distance_explored=None,
            topomap_lengths=topomap_lengths,
            batch_first_idx=batch_first_idx,
            repetition=repetition,
            num_agents=num_agents,
        )
        env.close_all()

    def log_info(
        self,
        save_dir: str,
        frames: List[List[Image.Image]],
        reached_node: Optional[List[int]],
        valid_until: List[int],
        topomap_lengths: Optional[List[int]],
        batch_first_idx: int,
        repetition: int,
        distance_explored: Optional[List[float]],
        num_agents: int,
        eval_type: str = "nav",
    ) -> None:
        for agent_id in tqdm.tqdm(range(num_agents), desc="Saving"):
            os.makedirs(f"{save_dir}/{(agent_id + batch_first_idx):04d}", exist_ok=True)
            # uncomment to log GIFs
            # frames[agent_id][0].save(
            #     f"{save_dir}/{(agent_id + batch_first_idx):04d}/{repetition}_navigation_{self.model_type}.gif",
            #     format="GIF",
            #     append_images=frames[agent_id][1 : valid_until[agent_id] + 2],
            #     save_all=True,
            #     duration=len(frames[agent_id][1 : valid_until[agent_id] + 2]),
            #     loop=0,
            # )
            if eval_type == "nav":
                assert reached_node is not None, "Reached node is None"
                assert topomap_lengths is not None, "Topomap lengths is None"
                myvar = {
                    "goal_reached": int(reached_node[agent_id]),
                    "max_goal": int(topomap_lengths[agent_id]),
                }
            elif eval_type == "exp":
                assert distance_explored is not None, "Distance explored is None"
                myvar = {
                    "distance_explored": float(distance_explored[agent_id]),
                    "steps_until_collision": int(valid_until[agent_id]),
                }
            else:
                raise ValueError(f"Unknown eval_type: {eval_type}")
            with open(
                f"{save_dir}/{(agent_id + batch_first_idx):04d}/{repetition}_{self.model_type}.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(myvar, f)

    def run_model_on_scene(
        self,
        scene_path: str,
        scene: str,
        result_path: str,
        model_cfg: str,
        model_ckpt: str,
        max_n_agents: int,
        device: str = "cuda:0",
        eval_type: str = "nav",
        controller: str = "velocity",
        gt_scale: float = 0.25,
    ) -> None:
        # get scene config and paths
        scene_config_path = f"{scene_path}/scene_config.yaml"
        scene_config = load_config(scene_config_path)
        glb_path = scene_config["glb_path"]
        num_agents = scene_config["num_agents"]

        # run in batches of max_n_agents
        for batch in range(0, num_agents, max_n_agents):
            click.echo(
                click.style(
                    f">> Processing scene {scene} with agents {batch} to {min(batch + max_n_agents, num_agents)}",
                    fg="magenta",
                    bold=True,
                )
            )

            # load agent configs and initialize env
            agent_ids = [
                id - batch for id in range(batch, min(batch + max_n_agents, num_agents))
            ]
            agent_config = {}
            pos_0, quat_0 = [], []
            cam_height, cam_xi, cam_alpha, cam_focal_length = [], [], [], []
            agent_positions = [[] for _ in range(num_agents)]
            topomap = [[] for _ in range(num_agents)]

            for agent_id in tqdm.tqdm(
                agent_ids,
                desc=f"Loading agents {batch} to {min(batch + max_n_agents, num_agents)}",
                colour="blue",
            ):
                # load agent pose
                agent_config_path = f"{scene_path}/{(agent_id + batch):04d}/config.npz"
                agent_config[agent_id] = np.load(agent_config_path, allow_pickle=True)
                pos_0.append(agent_config[agent_id]["positions"][0])
                quat_0.append(agent_config[agent_id]["quaternions"][0])
                if "info" in agent_config[agent_id]:
                    cam_height.append(
                        agent_config[agent_id]["info"].item()["camera_height"]
                    )
                    cam_xi.append(agent_config[agent_id]["info"].item()["fisheye_xi"])
                    cam_alpha.append(
                        agent_config[agent_id]["info"].item()["fisheye_alpha"]
                    )
                    cam_focal_length.append(
                        agent_config[agent_id]["info"].item()["focal_length"]
                    )
                else:
                    cam_height.append(0.5)
                    cam_xi.append(1.0)
                    cam_alpha.append(0.0)
                    cam_focal_length.append(364.8)
                agent_positions[agent_id] = agent_config[agent_id]["positions"]

                # read topomap images
                image_path = f"{scene_path}/{(agent_id + batch):04d}"
                image_files = [
                    f
                    for f in os.listdir(image_path)
                    if f.endswith(".png") and re.match(r"^\d+\.png$", f)
                ]
                image_files.sort(key=lambda x: int(x.split(".")[0]))
                for image in image_files:
                    topomap[agent_id].append(Image.open(f"{image_path}/{image}"))

            # create save directory
            save_dir = f"{result_path}/{scene}"
            os.makedirs(save_dir, exist_ok=True)

            # run repetitions
            for repetition in range(self.repetitions_per_trajectory):
                env = HabitatEnvMultiAgent(
                    scene_id_path=glb_path,
                    agent_ids=agent_ids,
                    camera_height=cam_height,
                    fisheye_xi=cam_xi,
                    fisheye_alpha=cam_alpha,
                    focal_length=cam_focal_length,
                )
                env.update_agent_states(pos_0, quat_0)
                if eval_type == "nav":
                    self.run_nav(
                        env=env,
                        agent_positions=agent_positions,
                        topomap=topomap,
                        save_dir=save_dir,
                        model_cfg=model_cfg,
                        model_ckpt=model_ckpt,
                        device=device,
                        controller=controller,
                        batch_first_idx=batch,
                        repetition=repetition,
                        gt_scale=gt_scale,
                    )
                elif eval_type == "exp":
                    self.run_exploration(
                        env=env,
                        save_dir=save_dir,
                        model_cfg=model_cfg,
                        model_ckpt=model_ckpt,
                        device=device,
                        controller=controller,
                        batch_first_idx=batch,
                        repetition=repetition,
                        gt_scale=gt_scale,
                    )
                else:
                    raise ValueError(f"Unknown eval_type: {eval_type}")
