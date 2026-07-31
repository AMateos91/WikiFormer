"""
utils.py

Utility functions for WikiFormer.

Includes:
- Random seed control
- Checkpoint management
- Model statistics
- Directory handling
"""


import os
import time
import random

import numpy as np
import torch



###############################################################
# Reproducibility
###############################################################


def set_seed(
    seed
):

    """
    Set random seeds for reproducible experiments.
    """


    random.seed(
        seed
    )


    np.random.seed(
        seed
    )


    torch.manual_seed(
        seed
    )


    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )



###############################################################
# Directory utilities
###############################################################


def create_directory(
    path
):

    """
    Create directory if missing.
    """


    os.makedirs(
        path,
        exist_ok=True
    )



###############################################################
# Model statistics
###############################################################


def count_parameters(
    model
):

    """
    Count trainable parameters.
    """


    return sum(

        parameter.numel()

        for parameter in model.parameters()

        if parameter.requires_grad

    )



def print_model_size(
    model
):

    params = count_parameters(
        model
    )


    if params >= 1_000_000:

        size = (
            params / 1_000_000
        )

        print(
            f"Model parameters: {size:.2f}M"
        )


    else:

        print(
            f"Model parameters: {params}"
        )



###############################################################
# Checkpoint saving
###############################################################


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    path,
    scheduler=None
):

    """
    Save training state.
    """


    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "loss": loss

    }


    if scheduler is not None:

        checkpoint[
            "scheduler_state_dict"
        ] = scheduler.state_dict()



    torch.save(
        checkpoint,
        path
    )



###############################################################
# Checkpoint loading
###############################################################


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    device="cpu"
):

    """
    Load training state.
    """


    checkpoint = torch.load(

        path,

        map_location=device

    )


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]

    )



    if optimizer is not None:

        optimizer.load_state_dict(

            checkpoint[
                "optimizer_state_dict"
            ]

        )



    if (
        scheduler is not None
        and
        "scheduler_state_dict"
        in checkpoint
    ):

        scheduler.load_state_dict(

            checkpoint[
                "scheduler_state_dict"
            ]

        )



    return checkpoint.get(
        "epoch",
        0
    ), checkpoint.get(
        "loss",
        None
    )



###############################################################
# Training timer
###############################################################


class Timer:


    def __init__(
        self
    ):

        self.start_time = None



    def start(
        self
    ):

        self.start_time = time.time()



    def elapsed(
        self
    ):

        if self.start_time is None:

            return 0


        return (
            time.time()
            -
            self.start_time
        )



###############################################################
# Gradient utilities
###############################################################


def clip_gradients(
    model,
    max_norm
):

    """
    Prevent exploding gradients.
    """


    torch.nn.utils.clip_grad_norm_(

        model.parameters(),

        max_norm

    )



###############################################################
# Device info
###############################################################


def device_info():

    """
    Print hardware information.
    """


    if torch.cuda.is_available():

        print(
            "CUDA available"
        )


        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )


        print(
            "VRAM:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                /
                1024**3,
                2
            ),
            "GB"
        )


    else:

        print(
            "Running on CPU"
        )



###############################################################
# Test
###############################################################


if __name__ == "__main__":

    set_seed(42)

    device_info()

    print(
        "Utils ready."
    )
