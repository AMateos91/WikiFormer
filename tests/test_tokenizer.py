from tokenizer import Tokenizer
from config import Config



def test_encode_decode():

    tokenizer = Tokenizer(
        Config
    )


    text = (
        "hello world"
    )


    tokens = tokenizer.encode(
        text
    )


    decoded = tokenizer.decode(
        tokens
    )


    assert len(tokens) > 0

    assert isinstance(
        decoded,
        str
    )
