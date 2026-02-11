import os
from typing import Dict

import click
import torch
import torch.nn as nn
import wandb
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from metricnet.training.evaluate import evaluate
from metricnet.training.train import train


def train_loop(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_loader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    epochs: int,
    device: torch.device,
    project_folder: str,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    current_epoch: int = 0,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
    eval_freq: int = 1,
) -> None:
    # create project folder if it doesn't exist
    if not os.path.exists(project_folder):
        os.makedirs(project_folder)
    # path to save latest model
    latest_path = os.path.join(project_folder, "latest.pth")

    # training loop
    for epoch in range(current_epoch, current_epoch + epochs):
        if train_model:
            click.echo(
                click.style(
                    f"> Start epoch {epoch}/{current_epoch + epochs - 1}",
                    fg="magenta",
                )
            )
            train(
                model=model,
                optimizer=optimizer,
                dataloader=train_loader,
                transform=transform,
                device=device,
                epoch=epoch,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                use_wandb=use_wandb,
            )
            lr_scheduler.step()

        # save model
        numbered_path = os.path.join(project_folder, f"{epoch}.pth")
        torch.save(model.state_dict(), numbered_path)
        torch.save(model.state_dict(), latest_path)
        numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
        latest_optimizer_path = os.path.join(project_folder, "optimizer_latest.pth")
        torch.save(optimizer.state_dict(), latest_optimizer_path)
        numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
        latest_scheduler_path = os.path.join(project_folder, "scheduler_latest.pth")
        torch.save(lr_scheduler.state_dict(), latest_scheduler_path)

        # evaluate
        if (epoch + 1) % eval_freq == 0:
            for dataset_type in test_dataloaders:
                click.echo(
                    click.style(
                        f"> Start {dataset_type} EVALUATION Epoch {epoch}/{current_epoch + epochs - 1}",
                        fg="cyan",
                    )
                )
                loader = test_dataloaders[dataset_type]
                evaluate(
                    model=model,
                    eval_type=dataset_type,
                    dataloader=loader,
                    transform=transform,
                    device=device,
                    epoch=epoch,
                    print_log_freq=print_log_freq,
                    wandb_log_freq=wandb_log_freq,
                    use_wandb=use_wandb,
                    eval_fraction=eval_fraction,
                )

        # log to wandb
        if use_wandb:
            wandb.log(
                {
                    "lr": optimizer.param_groups[0]["lr"],
                },
                commit=False,
            )
        # step the lr scheduler
        if lr_scheduler is not None:
            lr_scheduler.step()

    # flush wandb logs
    if use_wandb:
        wandb.log({})
