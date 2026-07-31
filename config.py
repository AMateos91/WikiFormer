"""
config.py

Central configuration for WikiFormer.

All project parameters are stored here:
- Model
- Training
- Dataset
- Tokenizer
- Generation
"""


import torch
import os



class Config:


    ###########################################################
    # Device
    ###########################################################

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    ###########################################################
    # Paths
    ###########################################################

    ROOT_DIR = os.path.dirname(
        os.path.abspath(__file__)
    )


    DATA_DIR = os.path.join(
        ROOT_DIR,
        "data"
    )


    CHECKPOINT_DIR = os.path.join(
        ROOT_DIR,
        "checkpoints"
    )

    RAW_DATA_DIR = os.path.join(
    ROOT_DIR,
    "raw"
    )

    TOKENIZER_DIR = os.path.join(
    ROOT_DIR,
    "tokenizer"
    )

    TOKENIZER_PATH = os.path.join(
    TOKENIZER_DIR,
    "wiki.model"
    )


    ###########################################################
    # Dataset
    ###########################################################

    TRAIN_FILE = os.path.join(
        DATA_DIR,
        "train.txt"
    )


    VAL_FILE = os.path.join(
        DATA_DIR,
        "validation.txt"
    )


    SEQUENCE_LENGTH = 256


    ###########################################################
    # Tokenizer
    ###########################################################

    VOCAB_SIZE = 32000


    ###########################################################
    # Model
    ###########################################################

    # vocabulary size
    # will normally match tokenizer

    VOCAB_SIZE = 32000


    EMBEDDING_DIM = 384


    NUM_LAYERS = 6


    NUM_HEADS = 6


    FF_DIM = 1536


    DROPOUT = 0.1



    ###########################################################
    # Training
    ###########################################################

    BATCH_SIZE = 32


    EPOCHS = 10


    LEARNING_RATE = 3e-4


    WEIGHT_DECAY = 0.01


    GRADIENT_CLIP = 1.0


    WARMUP_STEPS = 1000



    ###########################################################
    # Optimization
    ###########################################################

    USE_AMP = True


    USE_COMPILE = True



    ###########################################################
    # Checkpoints
    ###########################################################

    SAVE_EVERY = 1000


    LAST_CHECKPOINT = os.path.join(
        CHECKPOINT_DIR,
        "checkpoint_latest.pt"
    )



    ###########################################################
    # Generation
    ###########################################################

    MAX_NEW_TOKENS = 200


    TEMPERATURE = 0.8


    TOP_K = 40


    TOP_P = 0.95



    ###########################################################
    # Logging
    ###########################################################

    LOG_INTERVAL = 100



    ###########################################################
    # Random seed
    ###########################################################

    SEED = 42




###############################################################
# Utility
###############################################################


def print_config():

    """
    Print active configuration.
    """

    for key, value in Config.__dict__.items():

        if not key.startswith("_"):

            print(
                f"{key}: {value}"
            )



if __name__ == "__main__":

    print_config()
