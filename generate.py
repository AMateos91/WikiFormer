"""
generate.py

Text generation script for WikiFormer.

Features:
- Load trained checkpoint
- Temperature sampling
- Top-k sampling
- Top-p nucleus sampling
- CUDA support

Educational GPT-style inference pipeline.
"""


import torch
import torch.nn.functional as F

from config import Config
from model import GPTModel
from tokenizer import Tokenizer


###############################################################
# Sampling utilities
###############################################################


def top_k_filter(
    logits,
    k
):
    """
    Keep only the k most probable tokens.
    """

    values, _ = torch.topk(
        logits,
        k
    )

    minimum = values[:, -1]

    logits[
        logits < minimum
    ] = float("-inf")

    return logits



def top_p_filter(
    logits,
    p
):
    """
    Nucleus sampling.
    """

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


    sorted_logits[mask] = float("-inf")


    logits.scatter_(
        1,
        sorted_indices,
        sorted_logits
    )


    return logits



###############################################################
# Text generation
###############################################################


@torch.no_grad()
def generate(
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


    tokens = tokenizer.encode(
        prompt
    )


    input_ids = torch.tensor(
        tokens,
        dtype=torch.long
    ).unsqueeze(0)


    input_ids = input_ids.to(
        device
    )


    for _ in range(max_tokens):


        # Limit context length

        context = input_ids[
            :,
            -Config.SEQ_LENGTH:
        ]


        logits = model(
            context
        )


        logits = logits[:, -1, :]


        logits = logits / temperature



        if top_k > 0:

            logits = top_k_filter(
                logits,
                top_k
            )


        if top_p < 1.0:

            logits = top_p_filter(
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


        input_ids = torch.cat(
            [
                input_ids,
                next_token
            ],
            dim=1
        )



    output = tokenizer.decode(
        input_ids[0].tolist()
    )


    return output



###############################################################
# Load checkpoint
###############################################################


def load_model(
    checkpoint_path,
    device
):


    cfg = Config()


    model = GPTModel(
        cfg
    )


    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )


    model.load_state_dict(
        checkpoint["model"]
    )


    model.to(device)


    model.eval()


    return model



###############################################################
# Main
###############################################################


def main():


    cfg = Config()


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        "Using device:",
        device
    )


    tokenizer = Tokenizer(
        cfg
    )


    checkpoint = (
        "checkpoints/"
        "checkpoint_latest.pt"
    )


    model = load_model(
        checkpoint,
        device
    )


    prompt = input(
        "\nPrompt: "
    )


    result = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=cfg.MAX_NEW_TOKENS,
        temperature=cfg.TEMPERATURE,
        top_k=cfg.TOP_K,
        top_p=cfg.TOP_P,
        device=device
    )


    print(
        "\nGenerated text:\n"
    )

    print(result)



if __name__ == "__main__":

    main()
