# WikiFormer
A GPT-style Transformer implemented from scratch in PyTorch 2.x

WikiFormer is an educational project focused on understanding the internal components of modern language models:

- Tokenization
- Transformer architecture
- Self-attention
- Training pipeline
- Text generation


## Features

- PyTorch Transformer implementation
- Multi-head self attention
- Causal masking
- SentencePiece tokenizer
- Mixed precision training
- AdamW optimizer
- Checkpoint system
- Text generation with:
  - Temperature
  - Top-k sampling
  - Top-p sampling


## Project Structure

WikiFormer/

├── attention.py
├── config.py
├── dataset.py
├── generate.py
├── model.py
├── tokenizer.py
├── train.py
├── utils.py
│
├── data/
├── checkpoints/
└── tokenizer/

## Installation

Clone repository:

git clone https://github.com/AMateos91/WikiFormer.git

cd WikiFormer

Install dependencies:

pip install -r requirements.txt

## Preparing Data

Place your text corpus:

raw/wiki.txt

Then run:

python prepare_data.py

This creates:

data/train.txt and
data/validation.txt

## Training

Start training:

python train.py

The model will save checkpoints:

checkpoints/checkpoint_latest.pt

## Text Generation

After training:

python generate.py

## Prompt:

Artificial intelligence

- Output:
Artificial intelligence is a field...

## Architecture

The model contains:

-Token embeddings
-Positional embeddings
-Transformer blocks
-Multi-head attention
-Feed-forward layers
-Layer normalization

## License

MIT License

---

# Tests 

We have created:

tests/

├── test_model.py
└── test_tokenizer.py
