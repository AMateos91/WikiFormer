"""
train.py

Training pipeline for WikiFormer.

Features:
- PyTorch training loop
- Mixed Precision (AMP)
- AdamW optimizer
- Learning rate scheduler
- Validation
- Checkpointing
- Resume training
"""


import os
import math

import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from torch.amp import autocast, GradScaler


from config import Config
from dataset import create_dataloaders
from model import GPTModel

from utils import (
    set_seed,
    create_directory,
    save_checkpoint,
    load_checkpoint,
    print_model_size,
    clip_gradients,
    device_info
)



###############################################################
# Learning rate scheduler
###############################################################


def warmup_scheduler(
    optimizer,
    warmup_steps
):

    def lr_lambda(step):

        if step < warmup_steps:

            return float(step + 1) / float(
                warmup_steps
            )

        return 1.0


    return LambdaLR(
        optimizer,
        lr_lambda
    )



###############################################################
# Validation
###############################################################


def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0

    batches = 0


    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)


            with autocast(
                device_type=device.type
            ):

                logits = model(x)


                loss = criterion(

                    logits.reshape(
                        -1,
                        logits.size(-1)
                    ),

                    y.reshape(-1)

                )


            total_loss += loss.item()

            batches += 1


    return total_loss / max(
        batches,
        1
    )



###############################################################
# Training epoch
###############################################################


def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    criterion,
    device,
    epoch
):

    model.train()

    total_loss = 0

    batches = 0


    for step, (x, y) in enumerate(loader):


        x = x.to(
            device,
            non_blocking=True
        )


        y = y.to(
            device,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        with autocast(
            device_type=device.type,
            enabled=Config.USE_AMP
        ):


            logits = model(x)


            loss = criterion(

                logits.reshape(
                    -1,
                    logits.size(-1)
                ),

                y.reshape(-1)

            )


        scaler.scale(
            loss
        ).backward()



        scaler.unscale_(
            optimizer
        )


        clip_gradients(

            model,

            Config.GRADIENT_CLIP

        )


        scaler.step(
            optimizer
        )


        scaler.update()


        scheduler.step()


        total_loss += loss.item()

        batches += 1



        if step % Config.LOG_INTERVAL == 0:

            print(
                f"Epoch {epoch} "
                f"Step {step} "
                f"Loss {loss.item():.4f}"
            )


    return total_loss / max(
        batches,
        1
    )



###############################################################
# Main
###############################################################


def main():

    cfg = Config


    set_seed(
        cfg.SEED
    )


    create_directory(
        cfg.CHECKPOINT_DIR
    )


    device_info()


    device = torch.device(
        cfg.DEVICE
    )


    print(
        "\nLoading dataset..."
    )


    train_loader, val_loader = (
        create_dataloaders(cfg)
    )


    print(
        "Building model..."
    )


    model = GPTModel(
        cfg
    ).to(device)


    print_model_size(
        model
    )



    ###########################################################
    # Compile model
    ###########################################################

    if (
        cfg.USE_COMPILE
        and
        hasattr(torch, "compile")
    ):

        print(
            "Compiling model..."
        )

        model = torch.compile(
            model
        )



    ###########################################################
    # Optimizer
    ###########################################################

    optimizer = AdamW(

        model.parameters(),

        lr=cfg.LEARNING_RATE,

        weight_decay=cfg.WEIGHT_DECAY

    )


    scheduler = warmup_scheduler(

        optimizer,

        cfg.WARMUP_STEPS

    )


    criterion = nn.CrossEntropyLoss()


    scaler = GradScaler(
        enabled=cfg.USE_AMP
    )



    ###########################################################
    # Resume checkpoint
    ###########################################################

    start_epoch = 1


    if os.path.exists(
        cfg.LAST_CHECKPOINT
    ):

        print(
            "Loading checkpoint..."
        )


        epoch, loss = load_checkpoint(

            cfg.LAST_CHECKPOINT,

            model,

            optimizer,

            scheduler,

            device

        )


        start_epoch = epoch + 1


        print(
            f"Resumed from epoch {epoch}"
        )



    ###########################################################
    # Training loop
    ###########################################################

    for epoch in range(

        start_epoch,

        cfg.EPOCHS + 1

    ):


        print(
            f"\n===== Epoch {epoch}/{cfg.EPOCHS} ====="
        )


        train_loss = train_epoch(

            model,

            train_loader,

            optimizer,

            scheduler,

            scaler,

            criterion,

            device,

            epoch

        )


        val_loss = evaluate(

            model,

            val_loader,

            criterion,

            device

        )


        print(
            f"""
Epoch complete

Train loss:
{train_loss:.4f}

Validation loss:
{val_loss:.4f}
"""
        )



        save_checkpoint(

            model,

            optimizer,

            epoch,

            val_loss,

            cfg.LAST_CHECKPOINT,

            scheduler

        )


        print(
            "Checkpoint saved."
        )



if __name__ == "__main__":

    main()
