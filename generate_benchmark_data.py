import argparse
import multiprocessing
import os

import click
import numpy as np
import tqdm
import yaml

from metricnet.benchmark.environment import HabitatEnvMultiAgent


def process_scene(
    scenes_loc: str, num_agents: int, scene_name: str, output_dir: str
) -> None:
    # assert file exists
    scene_path = f"{scenes_loc}/{scene_name}"
    assert os.path.isdir(scene_path), f"Scene Directory {scene_name} does not exist"
    glb_path = f"{scene_path}/{scene_name.strip().split('-')[-1]}.basis.glb"
    assert os.path.exists(glb_path), f"GLB file in scene {scene_name} does not exist"

    # initialize environment
    scene_config = {"glb_path": glb_path, "num_agents": num_agents}
    agent_ids = list(range(num_agents))
    env = HabitatEnvMultiAgent(
        scene_id_path=glb_path,
        agent_ids=agent_ids,
        camera_height=np.random.uniform(0.25, 1.0, size=num_agents),
        fisheye_xi=np.random.uniform(0.0, 1.0, size=num_agents),
        fisheye_alpha=np.zeros(num_agents),
        focal_length=np.random.normal(364.8, 20.0, size=num_agents),
    )

    # log episode data
    cur_scene_loc = f"{output_dir}/{scene_name}"
    for agent_id in tqdm.tqdm(agent_ids, desc=f"Processing {scene_name}"):
        ep_loc = f"{cur_scene_loc}/{agent_id:04d}"
        os.makedirs(ep_loc, exist_ok=True)
        log_episode(env, agent_id, ep_loc)
    with open(f"{cur_scene_loc}/scene_config.yaml", "w") as f:
        yaml.safe_dump(scene_config, f)
    env.close_all()
    del env


def log_episode(env: HabitatEnvMultiAgent, agent_id: int, ep_loc: str) -> None:
    # collect topomap and save
    agent_states = env.collect_topomap(agent_id=agent_id, save_path=ep_loc)
    ep_config = {}
    positions = [np.array(state.position) for state in agent_states[agent_id]]
    quats = [
        np.array(
            [state.rotation.x, state.rotation.y, state.rotation.z, state.rotation.w]
        )
        for state in agent_states[agent_id]
    ]
    info = env.save_dataset_info(agent_id=agent_id)
    ep_config["positions"] = positions
    ep_config["quaternions"] = quats
    ep_config["info"] = info
    np.savez(f"{ep_loc}/config.npz", **ep_config)


if __name__ == "__main__":
    # set up paths and parse arguments
    parser = argparse.ArgumentParser(
        description="Habitat Sim Benchmark Data Generation"
    )
    parser.add_argument(
        "--scenes_loc",
        "-s",
        type=str,
        default="<ROOT>/mp3d/versioned_data/hm3d-0.2/hm3d/val",
        help="Path to the directory containing scene folders.",
    )
    parser.add_argument(
        "--num_agents",
        "-n",
        type=int,
        default=20,
        help="Number of agents to simulate per scene.",
    )
    parser.add_argument(
        "--num_workers",
        "-w",
        type=int,
        default=1,
        help="Number of workers to use for parallel processing.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="./benchmark",
        help="Directory to save benchmark outputs.",
    )
    args = parser.parse_args()

    # initialize
    scene_names = sorted(
        [
            x
            for x in os.listdir(args.scenes_loc)
            if os.path.isdir(os.path.join(args.scenes_loc, x))
        ]
    )
    click.echo(click.style(f"Found {len(scene_names)} scenes", fg="green"))
    agent_ids = [id for id in range(args.num_agents)]
    click.echo(
        click.style(
            f">> Using {min(args.num_workers, multiprocessing.cpu_count())} workers",
            fg="yellow",
        )
    )
    click.echo(
        click.style(f">> Each scene will have {args.num_agents} agents", fg="yellow")
    )
    mp_args = [
        (args.scenes_loc, args.num_agents, scene_name, args.output_dir)
        for scene_name in scene_names
    ]
    with multiprocessing.Pool(
        processes=min(args.num_workers, multiprocessing.cpu_count())
    ) as pool:
        pool.starmap(process_scene, mp_args)

