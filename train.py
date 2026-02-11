import argparse
import os
import time

import click
import numpy as np
import torch
import wandb
import yaml
from torchvision import transforms

from metricnet.data.dataset import Dataset
from metricnet.models.metricnet import MetricNet
from metricnet.training.loop import train_loop


def main(config: dict) -> None:
    # set up device
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif isinstance(config["gpu_ids"], int):
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["gpu_ids"]]
        )
        click.echo(
            click.style(f">> Using GPUs: {config['gpu_ids']}", fg="green", bold=True)
        )
    else:
        click.echo(click.style(">> No GPUs available, using CPU", fg="red", bold=True))
    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    # set seeds for reproducibility
    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # allow mixed precision
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # set up imagenet transform
    transform = transforms.Compose(
        [
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # load the data
    train_dataset = []
    test_dataloaders = {}
    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                dataset = Dataset(
                    data_folder=data_config["data_folder"],
                    data_split_folder=data_config[data_split_type],
                    dataset_name=dataset_name,
                    image_size=config["image_size"],
                    waypoint_spacing=data_config["waypoint_spacing"],
                    len_traj_pred=config["len_traj_pred"],
                )
                if data_split_type == "train":
                    train_dataset.append(dataset)
                else:
                    dataset_type = f"{dataset_name}_{data_split_type}"
                    if dataset_type not in test_dataloaders:
                        test_dataloaders[dataset_type] = {}
                    test_dataloaders[dataset_type] = dataset

    # combine all the datasets from different robots
    train_dataset = torch.utils.data.ConcatDataset(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
        persistent_workers=False,
    )
    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]

    for dataset_type, dataset in test_dataloaders.items():
        test_dataloaders[dataset_type] = torch.utils.data.DataLoader(
            dataset,
            batch_size=config["eval_batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            drop_last=True,
            persistent_workers=False,
        )
        click.echo(
            click.style(
                f">> Loaded {len(dataset)} test samples for {dataset_type}",
                fg="cyan",
                bold=True,
            )
        )

    # create the model
    model = torch.compile(
        MetricNet(
            d_model=config["embedding_size"],
            n_conv_res_blocks=config.get("n_conv_res_blocks", 1),
            n_lin_res_blocks=config.get("n_lin_res_blocks", 1),
        ),
        mode="reduce-overhead",
        fullgraph=False,
    )

    # load pretrained depth encoder weights
    checkpoint = torch.load(
        os.path.join(config["depth_encoder_weights"]),
        map_location=device,
    )
    updated_state_dict = {
        k.replace("pretrained.", ""): v
        for k, v in checkpoint.items()
        if "pretrained" in k
    }
    model.depth_encoder.load_state_dict(updated_state_dict, strict=True)
    model = model.to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]), fused=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )

    # load pretrained model if specified
    current_epoch = 0
    if "load_run" in config:
        load_project_folder = os.path.join("logs", config["load_run"])
        click.echo(
            click.style(
                f">> Loading pre-trained model from {load_project_folder}",
                fg="yellow",
            )
        )
        if os.path.isdir(load_project_folder):
            latest_path = os.path.join(load_project_folder, "latest.pth")
        elif os.path.isfile(load_project_folder):
            latest_path = load_project_folder
        else:
            click.echo(
                click.style(
                    f">> Could not find pre-trained model at {load_project_folder}",
                    fg="red",
                )
            )
            return
        latest_checkpoint = torch.load(latest_path)
        if "model" in latest_checkpoint:
            model.load_state_dict(latest_checkpoint["model"], strict=True)
        else:
            model.load_state_dict(latest_checkpoint, strict=True)
        if "epoch" in latest_checkpoint:
            current_epoch = latest_checkpoint["epoch"] + 1
        if "optimizer" in latest_checkpoint:
            optimizer.load_state_dict(latest_checkpoint["optimizer"].state_dict())
        if scheduler is not None and "scheduler" in latest_checkpoint:
            scheduler.load_state_dict(latest_checkpoint["scheduler"].state_dict())

    train_loop(
        train_model=config["train"],
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        train_loader=train_loader,
        test_dataloaders=test_dataloaders,
        transform=transform,
        epochs=config["epochs"],
        device=device,
        project_folder=config["project_folder"],
        print_log_freq=config["print_log_freq"],
        wandb_log_freq=config["wandb_log_freq"],
        current_epoch=current_epoch,
        use_wandb=config["use_wandb"],
        eval_fraction=config["eval_fraction"],
        eval_freq=config["eval_freq"],
    )
    click.echo(
        click.style(
            f">> Training completed. Model saved to {config['project_folder']}",
            fg="green",
            bold=True,
        )
    )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="metricnet/config/metricnet.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()

    # load the default config and update with user config
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    with open(f"{this_file_dir}/metricnet/config/metricnet.yaml", "r") as f:
        default_config = yaml.safe_load(f)
    config = default_config
    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)
    click.echo(click.style(f">> Using config file: {args.config}", fg="yellow"))

    # update the default config with user config
    config.update(user_config)
    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]
    )
    os.makedirs(config["project_folder"])

    # set up wandb
    if config["use_wandb"]:
        wandb.login()
        wandb.init(
            project=config["project_name"],
            entity=config["entity"],
            settings=wandb.Settings(start_method="fork"),
        )

        wandb.save(args.config, policy="now")
        wandb.run.name = config["run_name"]
        if wandb.run:
            wandb.config.update(config)

    # run the training
    main(config)
