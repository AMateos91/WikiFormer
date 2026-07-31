"""
train.py

Training loop for a small GPT-style Transformer.

Educational implementation:
- PyTorch 2.x
- Mixed precision
- Checkpointing
- Validation loop
"""

import os
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from config import Config
from dataset import TextDataset
from model import GPTModel


def save_checkpoint(model, optimizer, epoch, loss, path):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": loss,
    }

    torch.save(checkpoint, path)


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    criterion,
    device
):

    model.train()

    total_loss = 0

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        with autocast(
            device_type=device.type
        ):

            logits = model(x)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            total_loss += loss.item()

    return total_loss / len(loader)


def main():

    cfg = Config()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    train_dataset = TextDataset(
        cfg.train_file,
        cfg.sequence_length
    )

    val_dataset = TextDataset(
        cfg.val_file,
        cfg.sequence_length
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size
    )


    model = GPTModel(
        cfg
    ).to(device)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )


    criterion = torch.nn.CrossEntropyLoss()


    scaler = GradScaler()


    for epoch in range(cfg.epochs):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            criterion,
            device
        )


        val_loss = evaluate(
            model,
            val_loader,
            criterion,
            device
        )


        print(
            f"Epoch {epoch+1}: "
            f"train={train_loss:.4f} "
            f"val={val_loss:.4f}"
        )


        save_checkpoint(
            model,
            optimizer,
            epoch,
            val_loss,
            f"checkpoint_{epoch}.pt"
        )


if __name__ == "__main__":
    main()
