import argparse
import multiprocessing
import os

import click
import numpy as np
import traceback
import yaml

from metricnet.benchmark.environment import HabitatEnvMultiAgent


def collect_topomap(
    scene_id: str, folder_name: str, split: str, n_agents: int, root_folder: str
) -> None:
    env = HabitatEnvMultiAgent(
        scene_id_path=os.path.join(
            root_folder, scene_id, f"{scene_id.split('-')[-1]}.basis.glb"
        ),
        image_width=640,
        image_height=480,
        camera_height=np.random.uniform(0.25, 1.0, size=n_agents),
        fisheye_xi=np.random.uniform(0.0, 1.0, size=n_agents),
        fisheye_alpha=np.zeros(n_agents),
        focal_length=np.random.normal(364.8, 20.0, size=n_agents),
        agent_ids=list(range(n_agents)),
        fps=1,
        context_size=3,
        min_topomap_distance=7.5,
        max_distance_between_images_in_topomap=0.25,
    )

    assert split == "val" or split == "train", (
        f"Split {split} is not supported, only 'val' and 'train' are supported"
    )

    for i in range(env.n_agents):
        try:
            env.collect_topomap(
                agent_id=i,
                save_path=f"{folder_name}/{split}/{scene_id}_{i}",
            )
        except Exception as e:
            click.echo(
                click.style(
                    f"Error collecting topomap for agent {i} in scene {scene_id}: {e}",
                    fg="red",
                )
            )
            click.echo(traceback.format_exc())

        try:
            env.save_topomap(
                save_path=f"{folder_name}/{split}/{scene_id}_{i}",
                agent_id=i,
            )
            info = env.save_dataset_info(
                agent_id=i,
            )
            with open(
                os.path.join(f"{folder_name}/{split}/{scene_id}_{i}", "params.yaml"),
                "w",
            ) as f:
                yaml.dump(info, f, default_flow_style=False)
        except Exception as e:
            click.echo(
                click.style(
                    f"Error saving topomap for agent {i} in scene {scene_id}: {e}",
                    fg="red",
                )
            )
            click.echo(traceback.format_exc())

    env.close_all()

    del env
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Habitat Sim Training Data Generation")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of parallel workers to use for data generation",
    )
    parser.add_argument(
        "--n_agents",
        type=int,
        default=20,
        help="Number of agents to simulate per scene",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="./dataset",
        help="Directory to save dataset outputs.",
    )
    parser.add_argument(
        "--scenes_loc",
        "-s",
        type=str,
        default="<ROOT>/mp3d/versioned_data/hm3d-0.2/hm3d/",
        help="Path to the directory containing scene folders.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Dataset split to use, either 'train' or 'val'",
    )
    args = parser.parse_args()

    # set up paths and parse arguments
    scene_codes = sorted(
        [
            x
            for x in os.listdir(args.scenes_loc)
            if os.path.isdir(os.path.join(args.scenes_loc, x))
        ]
    )
    click.echo(click.style(f"Found {len(scene_codes)} scenes", fg="green"))
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
        (scene_id, args.output_dir, args.split, args.n_agents, args.scenes_loc)
        for scene_id in scene_codes
    ]
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        pool.starmap(collect_topomap, mp_args)
