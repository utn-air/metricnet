import copy
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torchdiffeq

from metricnet.benchmark.agent import NavAgent
from metricnet.models.metricnav.guide import PathGuide as MetricNavPathGuide
from metricnet.models.navidiffusor.guide import PathGuide
from metricnet.training.utils import get_action


def get_action_prediction(
    obs_cond: torch.Tensor,
    model_type: str,
    nav_agent: NavAgent,
    k_steps: int,
    context_queue: List[torch.Tensor],
    num_agents: int,
    obs_now: torch.Tensor,
    noisy_action: torch.Tensor,
    pathguide: List[Union[PathGuide, MetricNavPathGuide]],
    model_cfg: Dict[str, Any],
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    if "navidiffusor" in model_type:
        for id in range(num_agents):
            pathguide[id].get_cost_map_via_tsdf(context_queue[id * 4 + 3])
        naction = noisy_action
        nav_agent.noise_scheduler.set_timesteps(k_steps)
        for k in nav_agent.noise_scheduler.timesteps[:]:
            noise_pred = nav_agent.model(
                "noise_pred_net",
                sample=naction,
                timestep=k,
                global_cond=obs_cond,
            )
            naction = nav_agent.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction
            ).prev_sample
            if k <= 6:
                for id in range(num_agents):
                    grad, _ = pathguide[id].get_gradient(
                        naction[id].unsqueeze(0),
                        goal_pos=None,
                        scale_factor=0.4,
                    )
                    naction[id] -= grad.squeeze(0)

    elif "metricnav" in model_type:
        for id in range(num_agents):
            pathguide[id].get_cost_map_via_tsdf(context_queue[id * 4 + 3])
        noisy_action = torch.randn(
            (num_agents * 40, model_cfg["len_traj_pred"], 2), device=device
        )
        with torch.no_grad():
            naction = copy.deepcopy(noisy_action)
            nav_agent.noise_scheduler.set_timesteps(k_steps)
            # first run to get the best goal idx
            for k in nav_agent.noise_scheduler.timesteps[:]:
                noise_pred = nav_agent.model(
                    "noise_pred_net",
                    sample=naction,
                    timestep=k,
                    global_cond=torch.repeat_interleave(obs_cond, 40, dim=0),
                )
                naction = nav_agent.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=naction
                ).prev_sample
            scale_factor = nav_agent.metricNet(
                obs_img=obs_now.to(device),
                waypoint=get_action(naction),
                repeat_features=True,
            )
            naction = (
                get_action(naction).detach().cpu().numpy()
                * (scale_factor.detach().cpu().numpy() / 1000)[:, None, None]
            )
            goal_positions = naction.reshape(
                num_agents, 40, model_cfg["len_traj_pred"], 2
            )[:, :, 5, :]
            best_goal_idx = []
            for id in range(num_agents):
                bg_idx, _ = pathguide[id].choose_goal_spherical(goal_positions[id])
                best_goal_idx.append(bg_idx + id * 40)
        # second run with path guidance
        naction = copy.deepcopy(noisy_action)
        nav_agent.noise_scheduler.set_timesteps(k_steps)  # just to be sure
        for k in nav_agent.noise_scheduler.timesteps[:]:
            noise_pred = nav_agent.model(
                "noise_pred_net",
                sample=naction,
                timestep=k,
                global_cond=torch.repeat_interleave(obs_cond, 40, dim=0),
            )
            naction = nav_agent.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction
            ).prev_sample
            if k <= 2:
                scale_factor = nav_agent.metricNet(
                    obs_img=obs_now.to(device),
                    waypoint=get_action(naction),
                    repeat_features=True,
                )
                for id in range(num_agents):
                    grad = pathguide[id].get_gradient_goal_collision(
                        naction.reshape(num_agents, 40, model_cfg["len_traj_pred"], 2)[
                            id
                        ],
                        goal_pos=goal_positions[id, best_goal_idx[id] - id * 40, :],
                        scale_factor=(scale_factor.reshape(num_agents, 40)[id] / 1000)[
                            :, None, None
                        ],
                        k=k,
                        alpha=0.5,
                        beta=0.01,
                    )
                    naction[id * 40 : (id + 1) * 40] -= grad

    elif "nomad" in model_type:
        with torch.no_grad():
            naction = noisy_action
            nav_agent.noise_scheduler.set_timesteps(k_steps)
            for k in nav_agent.noise_scheduler.timesteps[:]:
                noise_pred = nav_agent.model(
                    "noise_pred_net",
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond,
                )
                naction = nav_agent.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=naction
                ).prev_sample

    elif "flownav" in model_type:
        with torch.no_grad():
            traj = torchdiffeq.odeint(
                lambda t, x: nav_agent.model.forward(
                    "noise_pred_net", sample=x, timestep=t, global_cond=obs_cond
                ),
                noisy_action,
                torch.linspace(0, 1, k_steps, device=device),
                atol=1e-4,
                rtol=1e-4,
                method="euler",
            )
            naction = traj[-1]
    else:
        raise ValueError(f"Model type {model_type} not supported")

    return {
        "n_action": naction,
        "goal_positions": goal_positions if "metricnav" in model_type else None,
        "best_goal_idx": best_goal_idx if "metricnav" in model_type else None,
    }


def get_scaled_trajectory(
    model_type: str,
    naction: torch.Tensor,
    nav_agent: NavAgent,
    num_agents: int,
    model_cfg: Dict[str, Any],
    obs_now: torch.Tensor,
    pathguide: List[Union[PathGuide, MetricNavPathGuide]],
    device: torch.device = torch.device("cpu"),
    gt_scale: float = 0.26,
    goal_positions: torch.Tensor = None,
    best_goal_idx: List[int] = None,
) -> torch.Tensor:
    with torch.no_grad():
        if "metricnav" in model_type:
            scale_factor = nav_agent.metricNet(
                obs_img=obs_now.to(device),
                waypoint=get_action(naction),
                repeat_features=True,
            )
            naction = (
                get_action(naction).detach().cpu().numpy()
                * (scale_factor.detach().cpu().numpy() / 1000)[:, None, None]
            )
            best_cost_idx = []
            for id in range(num_agents):
                n_ids, _ = pathguide[id].tsdf_cost_map.Pos2Ind(
                    torch.tensor(naction[id * 40 : (id + 1) * 40])
                )
                dists_to_obs = (
                    pathguide[id].collision_cost_checker(n_ids).detach().cpu().numpy()
                )
                trajs_end_positions = naction.reshape(
                    num_agents, 40, model_cfg["len_traj_pred"], 2
                )[:, :, 5, :][id]
                goal_pos = goal_positions[id, best_goal_idx[id] - id * 40, :]
                dot_products = np.dot(trajs_end_positions, goal_pos)
                goal_norm_vector = np.linalg.norm(goal_pos)
                traj_norm_matrix = np.linalg.norm(trajs_end_positions, axis=1)
                denominator = goal_norm_vector * traj_norm_matrix + 1e-8
                goal_similarity = dot_products / denominator
                goal_similarity = (goal_similarity - goal_similarity.min()) / (
                    goal_similarity.max() - goal_similarity.min() + 1e-8
                )

                collision_avoidance = (dists_to_obs - dists_to_obs.min()) / (
                    dists_to_obs.max() - dists_to_obs.min() + 1e-8
                )
                combined_cost = 0.5 * goal_similarity + (1 - 0.5) * collision_avoidance
                best_cost_idx.append(np.argmax(combined_cost, axis=0).item() + 40 * id)
            naction = naction[best_cost_idx, :, :]
        else:
            naction = get_action(naction)
            if model_cfg.get("scale", {}).get("enabled", False):
                action_scaling_factor = (
                    nav_agent.metricNet(
                        obs_img=obs_now.to(device),
                        waypoint=naction,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    / 1000
                )
                if action_scaling_factor.ndim == 0:
                    action_scaling_factor = np.expand_dims(
                        action_scaling_factor, axis=0
                    )
            else:
                action_scaling_factor = gt_scale * np.ones(num_agents)

            naction = (
                naction.detach().cpu().numpy() * action_scaling_factor[:, None, None]
            )

    return naction
