"""
generate.py

Inference pipeline for WikiFormer.

Features:
- Load trained checkpoints
- Text generation
- Temperature sampling
- Top-k sampling
- Top-p nucleus sampling
- CUDA support
"""


import torch
import torch.nn.functional as F


from config import Config
from model import GPTModel
from tokenizer import Tokenizer



###############################################################
# Sampling functions
###############################################################


def apply_top_k(
    logits,
    k
):

    """
    Keep only the k most probable tokens.
    """

    if k <= 0:
        return logits


    values, _ = torch.topk(
        logits,
        k
    )


    minimum = values[:, -1].unsqueeze(
        -1
    )


    logits = torch.where(

        logits < minimum,

        torch.full_like(
            logits,
            float("-inf")
        ),

        logits

    )


    return logits




def apply_top_p(
    logits,
    p
):

    """
    Nucleus sampling.
    """

    if p >= 1.0:
        return logits


    sorted_logits, sorted_indices = torch.sort(

        logits,

        descending=True

    )


    probabilities = torch.softmax(

        sorted_logits,

        dim=-1

    )


    cumulative = torch.cumsum(

        probabilities,

        dim=-1

    )


    mask = cumulative > p


    mask[:, 1:] = mask[:, :-1]

    mask[:, 0] = False



    sorted_logits[mask] = float(
        "-inf"
    )


    logits.scatter_(

        1,

        sorted_indices,

        sorted_logits

    )


    return logits



###############################################################
# Generation
###############################################################


@torch.no_grad()
def generate_text(

    model,

    tokenizer,

    prompt,

    max_tokens,

    temperature,

    top_k,

    top_p,

    device

):


    model.eval()



    encoded = tokenizer.encode(

        prompt

    )


    tokens = torch.tensor(

        encoded,

        dtype=torch.long

    ).unsqueeze(0).to(device)



    for _ in range(max_tokens):


        context = tokens[

            :,

            -Config.SEQUENCE_LENGTH:

        ]



        logits = model(

            context

        )


        logits = logits[:, -1, :]



        logits /= temperature



        logits = apply_top_k(

            logits,

            top_k

        )


        logits = apply_top_p(

            logits,

            top_p

        )



        probabilities = torch.softmax(

            logits,

            dim=-1

        )


        next_token = torch.multinomial(

            probabilities,

            num_samples=1

        )



        tokens = torch.cat(

            [

                tokens,

                next_token

            ],

            dim=1

        )



    output = tokenizer.decode(

        tokens[0].tolist()

    )


    return output



###############################################################
# Checkpoint loader
###############################################################


def load_model(

    checkpoint_path,

    device

):


    model = GPTModel(
        Config
    )



    checkpoint = torch.load(

        checkpoint_path,

        map_location=device

    )



    state_dict = checkpoint[

        "model_state_dict"

    ]



    ###########################################################
    # Compatibility with torch.compile()
    ###########################################################

    cleaned_state_dict = {}



    for key, value in state_dict.items():


        if key.startswith(
            "_orig_mod."
        ):

            key = key.replace(

                "_orig_mod.",

                ""

            )


        cleaned_state_dict[key] = value



    model.load_state_dict(

        cleaned_state_dict,

        strict=True

    )



    model.to(device)


    model.eval()


    return model



###############################################################
# Main
###############################################################


def main():


    cfg = Config



    device = torch.device(

        cfg.DEVICE

    )


    print(

        "Device:",

        device

    )



    tokenizer = Tokenizer(

        cfg

    )



    model = load_model(

        cfg.LAST_CHECKPOINT,

        device

    )



    prompt = input(

        "\nPrompt: "

    )



    text = generate_text(

        model,

        tokenizer,

        prompt,

        cfg.MAX_NEW_TOKENS,

        cfg.TEMPERATURE,

        cfg.TOP_K,

        cfg.TOP_P,

        device

    )



    print(

        "\nGenerated text:\n"

    )


    print(text)



if __name__ == "__main__":

    main()
