import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from metricnet.training.logger import Logger


def train(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    epoch: int,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    use_wandb: bool = True,
):
    # set model to training mode
    model.train()

    # get number of batches
    num_batches = len(dataloader)

    # set up loggers
    metricnet_logger = Logger("metricnet_loss", "train", window_size=print_log_freq)
    loggers = {
        "metricnet_loss": metricnet_logger,
    }

    # training loop
    scaler = torch.amp.GradScaler(device=device)
    with tqdm.tqdm(dataloader, desc="Training", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_images,
                dataset_spacing,
                actions,
            ) = data

            # mixed precision training
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                batch_obs_images = transform(obs_images).to(
                    device, memory_format=torch.channels_last
                )
                pred = model(
                    obs_img=batch_obs_images,
                    waypoint=actions.to(device),
                )
                loss = F.mse_loss(
                    pred.squeeze(-1), dataset_spacing.float().to(device) * 1000
                )

            # backpropagation
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(retain_graph=False)
            scaler.step(optimizer)
            scaler.update()

            # logging
            if use_wandb:
                wandb.log({"metricnet_loss (train)": loss.item()})
            if i % print_log_freq == 0:
                losses = {"metricnet_loss": loss}
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(
                            f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}"
                        )
                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)
