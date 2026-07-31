"""
prepare_data.py

Prepare text files for WikiFormer training.

Creates:
- data/train.txt
- data/validation.txt
"""


import os
import random
from pathlib import Path

INPUT_DIR = "raw"

OUTPUT_DIR = "data"

TRAIN_FILE = os.path.join(
    OUTPUT_DIR,
    "train.txt"
)

VAL_FILE = os.path.join(
    OUTPUT_DIR,
    "validation.txt"
)


VALIDATION_SPLIT = 0.05

SEED = 42



def clean_line(
    text
):

    text = text.strip()

    while "  " in text:

        text = text.replace(
            "  ",
            " "
        )

    return text



def main():

    random.seed(
        SEED
    )


    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )



    print(
        "Loading dataset..."
    )


lines = []

files = sorted(Path(INPUT_DIR).glob("*.txt"))

if not files:
    raise FileNotFoundError(
        f"No .txt files found inside '{INPUT_DIR}'"
    )

for file in files:

    print(f"Reading {file.name}")

    with open(file, "r", encoding="utf-8") as f:

        for line in f:

            line = clean_line(line)

            if len(line) > 50:

                lines.append(line)



    print(
        "Documents:",
        len(lines)
    )



    random.shuffle(
        lines
    )



    split = int(

        len(lines)
        *
        (1 - VALIDATION_SPLIT)

    )


    train = lines[:split]

    validation = lines[split:]



    with open(

        TRAIN_FILE,

        "w",

        encoding="utf-8"

    ) as f:

        f.write(
            "\n".join(train)
        )



    with open(

        VAL_FILE,

        "w",

        encoding="utf-8"

    ) as f:

        f.write(
            "\n".join(validation)
        )



    print(
        "Dataset ready."
    )



if __name__ == "__main__":

    main()
