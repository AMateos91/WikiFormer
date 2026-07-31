"""
dataset.py

Dataset utilities for WikiFormer.

Responsibilities:
- Load text files
- Tokenize text
- Create input/target sequences
- Provide PyTorch Dataset objects
"""


import os

import torch

from torch.utils.data import Dataset, DataLoader

from tokenizer import Tokenizer



###############################################################
# Text loader
###############################################################


def load_text(
    path
):

    """
    Load plain text dataset.
    """

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )


    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()


    return text



###############################################################
# Token Dataset
###############################################################


class TextDataset(Dataset):


    def __init__(
        self,
        text,
        tokenizer,
        sequence_length
    ):

        self.sequence_length = (
            sequence_length
        )


        self.tokens = tokenizer.encode(
            text
        )


    def __len__(self):

        return (
            len(self.tokens)
            -
            self.sequence_length
        )



    def __getitem__(
        self,
        index
    ):


        x = self.tokens[
            index:
            index + self.sequence_length
        ]


        y = self.tokens[
            index + 1:
            index + self.sequence_length + 1
        ]


        return (

            torch.tensor(
                x,
                dtype=torch.long
            ),

            torch.tensor(
                y,
                dtype=torch.long
            )

        )



###############################################################
# Dataset builder
###############################################################


def create_dataloaders(
    config
):


    tokenizer = Tokenizer(
        config
    )


    train_text = load_text(
        config.TRAIN_FILE
    )


    val_text = load_text(
        config.VAL_FILE
    )


    train_dataset = TextDataset(

        train_text,

        tokenizer,

        config.SEQUENCE_LENGTH

    )


    val_dataset = TextDataset(

        val_text,

        tokenizer,

        config.SEQUENCE_LENGTH

    )



    train_loader = DataLoader(

        train_dataset,

        batch_size=config.BATCH_SIZE,

        shuffle=True,

        pin_memory=True

    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=config.BATCH_SIZE,

        shuffle=False,

        pin_memory=True

    )


    return (

        train_loader,

        val_loader

    )



###############################################################
# Quick test
###############################################################


if __name__ == "__main__":


    from config import Config


    train_loader, val_loader = (
        create_dataloaders(Config)
    )


    x, y = next(
        iter(train_loader)
    )


    print(
        "Input shape:",
        x.shape
    )


    print(
        "Target shape:",
        y.shape
    )
