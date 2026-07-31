"""
tokenizer.py

Tokenizer wrapper for WikiFormer.

Uses SentencePiece BPE tokenizer.

Responsibilities:
- Train tokenizer
- Load tokenizer
- Encode text
- Decode tokens
"""


import os
import sentencepiece as spm
from tqdm import tqdm



class Tokenizer:


    def __init__(
        self,
        config
    ):

        self.config = config

        self.model_file = (
            config.TOKENIZER_PATH
        )

        self.processor = spm.SentencePieceProcessor()


        if not os.path.exists(
            self.model_file
        ):

            raise FileNotFoundError(
                f"""
Tokenizer model not found:

{self.model_file}

Train the tokenizer first.
"""
            )


        self.processor.load(
            self.model_file
        )


        self.vocab_size = (
            self.processor.vocab_size()
        )



    ###########################################################
    # Encoding
    ###########################################################

    def encode(
        self,
        text
    ):

        """
        Convert text into token ids.
        """


        tokens = self.processor.encode(
            text,
            out_type=int
        )


        return tokens



    ###########################################################
    # Decoding
    ###########################################################

    def decode(
        self,
        tokens
    ):

        """
        Convert token ids back to text.
        """


        text = self.processor.decode(
            tokens
        )


        return text



    ###########################################################
    # Batch encoding
    ###########################################################

    def encode_batch(
        self,
        texts
    ):

        return [

            self.encode(text)

            for text in texts

        ]



    ###########################################################
    # Batch decoding
    ###########################################################

    def decode_batch(
        self,
        batches
    ):

        return [

            self.decode(tokens)

            for tokens in batches

        ]



    ###########################################################
    # Vocabulary
    ###########################################################

    def __len__(self):

        return self.vocab_size




###############################################################
# Tokenizer trainer
###############################################################


def train_tokenizer(
    input_dir="raw",
    output_prefix="tokenizer/wiki",
    vocab_size=32000
):

    """
    Train a SentencePiece BPE tokenizer.

    Parameters:

    input_dir:
        Any volume of text and files (Wikipedia).

    output_prefix:
        Output model prefix.

    vocab_size:
        Vocabulary size.
    """


    print(
        "Training tokenizer..."
    )


    spm.SentencePieceTrainer.train(

        input=input_file,

        model_prefix=output_prefix,

        vocab_size=vocab_size,

        model_type="bpe",

        character_coverage=1.0,

        shuffle_input_sentence=True,

        normalization_rule_name="nmt_nfkc",

        pad_id=0,

        unk_id=1,

        bos_id=2,

        eos_id=3,

        user_defined_symbols=[
            "<mask>"
        ]

    )


    print(
        "Tokenizer created:"
    )

    print(
        output_prefix + ".model"
    )



###############################################################
# Utility
###############################################################


def create_training_file(
    texts,
    output_file
):

    """
    Create a text file suitable
    for SentencePiece training.
    """


    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:


        for text in tqdm(texts):

            text = text.strip()


            if len(text) > 0:

                f.write(
                    text
                )

                f.write(
                    "\n"
                )



###############################################################
# Test
###############################################################


if __name__ == "__main__":


    class TestConfig:

        TOKENIZER_PATH = (
            "tokenizer/wiki.model"
        )


    tokenizer = Tokenizer(
        TestConfig()
    )


    sentence = (
        "Artificial intelligence "
        "is changing the world."
    )


    encoded = tokenizer.encode(
        sentence
    )


    print(
        "Tokens:"
    )

    print(
        encoded
    )


    decoded = tokenizer.decode(
        encoded
    )


    print(
        "\nDecoded:"
    )

    print(
        decoded
    )
