"""
model.py

GPT-style Transformer model.

Contains:
- Feed Forward Network
- Transformer Block
- GPT Model
"""

import torch
import torch.nn as nn

from attention import (
    MultiHeadSelfAttention,
    causal_mask
)



###############################################################
# Feed Forward Network
###############################################################


class FeedForward(nn.Module):


    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        dropout=0.1
    ):

        super().__init__()


        self.network = nn.Sequential(

            nn.Linear(
                embedding_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                embedding_dim
            ),

            nn.Dropout(
                dropout
            )

        )



    def forward(
        self,
        x
    ):

        return self.network(x)





###############################################################
# Transformer Block
###############################################################


class TransformerBlock(nn.Module):


    def __init__(
        self,
        embedding_dim,
        num_heads,
        ff_dim,
        dropout
    ):

        super().__init__()



        self.layer_norm1 = nn.LayerNorm(
            embedding_dim
        )


        self.layer_norm2 = nn.LayerNorm(
            embedding_dim
        )



        self.attention = MultiHeadSelfAttention(

            embedding_dim,

            num_heads,

            dropout

        )



        self.feed_forward = FeedForward(

            embedding_dim,

            ff_dim,

            dropout

        )



    def forward(
        self,
        x,
        mask
    ):


        #######################################################
        # Attention + residual
        #######################################################

        normalized = self.layer_norm1(
            x
        )


        attention_output, _ = self.attention(

            normalized,

            mask

        )


        x = x + attention_output



        #######################################################
        # Feed Forward + residual
        #######################################################

        normalized = self.layer_norm2(
            x
        )


        ff_output = self.feed_forward(
            normalized
        )


        x = x + ff_output


        return x





###############################################################
# GPT Model
###############################################################


class GPTModel(nn.Module):


    def __init__(
        self,
        config
    ):

        super().__init__()



        self.config = config



        #######################################################
        # Token embedding
        #######################################################

        self.token_embedding = nn.Embedding(

            config.VOCAB_SIZE,

            config.EMBEDDING_DIM

        )



        #######################################################
        # Position embedding
        #######################################################

        self.position_embedding = nn.Embedding(

            config.SEQUENCE_LENGTH,

            config.EMBEDDING_DIM

        )



        self.dropout = nn.Dropout(
            config.DROPOUT
        )



        #######################################################
        # Transformer layers
        #######################################################

        self.layers = nn.ModuleList(

            [

                TransformerBlock(

                    embedding_dim=config.EMBEDDING_DIM,

                    num_heads=config.NUM_HEADS,

                    ff_dim=config.FF_DIM,

                    dropout=config.DROPOUT

                )

                for _ in range(
                    config.NUM_LAYERS
                )

            ]

        )



        #######################################################
        # Final normalization
        #######################################################

        self.final_norm = nn.LayerNorm(

            config.EMBEDDING_DIM

        )



        #######################################################
        # Output projection
        #######################################################

        self.output = nn.Linear(

            config.EMBEDDING_DIM,

            config.VOCAB_SIZE,

            bias=False

        )



        #######################################################
        # Weight tying
        #######################################################

        self.output.weight = (
            self.token_embedding.weight
        )





    ###########################################################
    # Forward
    ###########################################################

    def forward(
        self,
        tokens
    ):


        batch, sequence = tokens.shape



        #######################################################
        # Embeddings
        #######################################################

        positions = torch.arange(

            sequence,

            device=tokens.device

        )


        positions = positions.unsqueeze(
            0
        )



        token_embeddings = self.token_embedding(
            tokens
        )


        position_embeddings = self.position_embedding(
            positions
        )



        x = (
            token_embeddings
            +
            position_embeddings
        )


        x = self.dropout(
            x
        )



        #######################################################
        # Causal attention mask
        #######################################################

        mask = causal_mask(

            sequence,

            tokens.device

        )



        #######################################################
        # Transformer stack
        #######################################################

        for layer in self.layers:

            x = layer(

                x,

                mask

            )



        #######################################################
        # Output logits
        #######################################################

        x = self.final_norm(
            x
        )


        logits = self.output(
            x
        )


        return logits





###############################################################
# Test
###############################################################


if __name__ == "__main__":


    from config import Config


    model = GPTModel(
        Config
    )


    sample = torch.randint(

        0,

        Config.VOCAB_SIZE,

        (

            2,

            Config.SEQUENCE_LENGTH

        )

    )


    output = model(
        sample
    )


    print(
        "Logits:",
        output.shape
    )
