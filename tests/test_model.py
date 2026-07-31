import torch

from config import Config
from model import GPTModel



def test_model_forward():

    model = GPTModel(
        Config
    )


    tokens = torch.randint(

        0,

        Config.VOCAB_SIZE,

        (
            2,
            Config.SEQUENCE_LENGTH
        )

    )


    output = model(
        tokens
    )


    assert output.shape == (

        2,

        Config.SEQUENCE_LENGTH,

        Config.VOCAB_SIZE

    )
