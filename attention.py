"""
attention.py

Transformer attention modules.

Contains:
- Scaled Dot Product Attention
- Multi Head Self Attention
- Causal masking
"""


import math

import torch
import torch.nn as nn
import torch.nn.functional as F



###############################################################
# Scaled Dot Product Attention
###############################################################


class ScaledDotProductAttention(nn.Module):


    def __init__(
        self,
        dropout=0.1
    ):

        super().__init__()

        self.dropout = nn.Dropout(
            dropout
        )



    def forward(
        self,
        query,
        key,
        value,
        mask=None
    ):

        """
        Attention formula:

        Attention(Q,K,V)
        =
        softmax(QK^T / sqrt(dk))V

        """


        dimension = (
            query.size(-1)
        )


        scores = torch.matmul(

            query,

            key.transpose(
                -2,
                -1
            )

        )


        scores = scores / math.sqrt(
            dimension
        )



        if mask is not None:

            scores = scores.masked_fill(

                mask == 0,

                float("-inf")

            )



        attention = F.softmax(

            scores,

            dim=-1

        )


        attention = self.dropout(
            attention
        )


        output = torch.matmul(

            attention,

            value

        )


        return output, attention





###############################################################
# Multi Head Self Attention
###############################################################


class MultiHeadSelfAttention(nn.Module):


    def __init__(
        self,
        embedding_dim,
        num_heads,
        dropout=0.1
    ):

        super().__init__()


        if embedding_dim % num_heads != 0:

            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )



        self.embedding_dim = (
            embedding_dim
        )


        self.num_heads = (
            num_heads
        )


        self.head_dim = (
            embedding_dim // num_heads
        )



        self.query = nn.Linear(

            embedding_dim,

            embedding_dim

        )


        self.key = nn.Linear(

            embedding_dim,

            embedding_dim

        )


        self.value = nn.Linear(

            embedding_dim,

            embedding_dim

        )



        self.output = nn.Linear(

            embedding_dim,

            embedding_dim

        )



        self.attention = (
            ScaledDotProductAttention(
                dropout
            )
        )



        self.dropout = nn.Dropout(
            dropout
        )



    ###########################################################
    # Split heads
    ###########################################################

    def split_heads(
        self,
        x
    ):

        """
        Convert:

        (batch, seq, embedding)

        into:

        (batch, heads, seq, head_dim)

        """


        batch, seq, _ = x.shape


        x = x.view(

            batch,

            seq,

            self.num_heads,

            self.head_dim

        )


        x = x.transpose(
            1,
            2
        )


        return x



    ###########################################################
    # Merge heads
    ###########################################################

    def merge_heads(
        self,
        x
    ):


        """
        Convert:

        (batch, heads, seq, head_dim)

        back into:

        (batch, seq, embedding)

        """


        batch, heads, seq, dim = (
            x.shape
        )


        x = x.transpose(
            1,
            2
        )


        x = x.contiguous().view(

            batch,

            seq,

            heads * dim

        )


        return x



    ###########################################################
    # Forward
    ###########################################################

    def forward(
        self,
        x,
        mask=None
    ):


        q = self.split_heads(

            self.query(x)

        )


        k = self.split_heads(

            self.key(x)

        )


        v = self.split_heads(

            self.value(x)

        )



        context, weights = self.attention(

            q,

            k,

            v,

            mask

        )



        context = self.merge_heads(
            context
        )


        output = self.output(
            context
        )


        output = self.dropout(
            output
        )


        return output, weights





###############################################################
# Causal Mask
###############################################################


def causal_mask(
    sequence_length,
    device
):

    """
    Creates lower triangular mask.

    Example:

    1 0 0 0
    1 1 0 0
    1 1 1 0
    1 1 1 1

    """


    mask = torch.tril(

        torch.ones(

            sequence_length,

            sequence_length,

            device=device

        )

    )


    return mask.unsqueeze(
        0
    ).unsqueeze(
        0
    )





###############################################################
# Test
###############################################################


if __name__ == "__main__":


    batch = 2

    seq = 8

    dim = 384


    x = torch.randn(

        batch,

        seq,

        dim

    )


    attention = MultiHeadSelfAttention(

        embedding_dim=dim,

        num_heads=6

    )


    mask = causal_mask(

        seq,

        x.device

    )


    output, weights = attention(

        x,

        mask

    )


    print(
        "Output:",
        output.shape
    )


    print(
        "Attention:",
        weights.shape
    )
