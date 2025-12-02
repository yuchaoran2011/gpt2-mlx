# GPT-2 Implementation with MLX

A minimal GPT-2 implementation using Apple's MLX framework, inspired by [Andrej Karpathy's NanoGPT](https://github.com/karpathy/nanoGPT).

This project implements a GPT-2 transformer model with training and inference capabilities, optimized for Apple Silicon using MLX's unified memory architecture. I documented learnings and pitfalls encountered during development in [Learnings from Implementing GPT-2 from Scratch using MLX](./blogpost.md). 

## Features

- GPT-2 transformer architecture with causal self-attention
- Training loop with gradient accumulation, learning rate scheduling, and weight decay
- Text generation with temperature and top-k sampling
- Optimized for Apple Silicon using MLX

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your training data:
   - Place your training text in `input.txt` (or modify `dataloader.py` to use a different file)

## Training

Run the training script:
```bash
python train.py
```

The training script will:
- Load and tokenize text from `input.txt`
- Train the model with gradient accumulation
- Generate sample text after training

You can modify training hyperparameters (learning rate, batch size, sequence length, etc.) directly in `train.py`.

## Inference

The `inference.py` module provides a `generate_text()` function that can be used to generate text from a trained model. The training script includes a sample generation at the end.

## Project Structure

- `gpt2.py` - GPT-2 model architecture
- `train.py` - Training script
- `inference.py` - Text generation utilities
- `dataloader.py` - Data loading and batching
- `checkpoint.py` - Model checkpointing utilities

