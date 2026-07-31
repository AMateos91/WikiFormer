"""
prepare_data.py

Prepare text files for WikiFormer training.

Creates:
- data/train.txt
- data/validation.txt
"""


import os
import random


INPUT_FILE = "raw/wiki.txt"

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


    with open(

        INPUT_FILE,

        "r",

        encoding="utf-8"

    ) as f:

        lines = [

            clean_line(line)

            for line in f

            if len(
                line.strip()
            ) > 50

        ]



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
