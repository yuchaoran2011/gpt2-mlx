from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_heads == 0

        self.n_heads = config.n_heads
        self.n_embed = config.n_embed

        self.head_size = config.n_embed // config.n_heads

        # 2nd dimension will be 3 * n_embed to get q, k, v in one go
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

        # Use k=1 to get the upper triangular part above the main diagonal
        # By default k = 0 includes the main diagonal
        self.mask = mx.triu(mx.ones((1, 1, config.block_size, config.block_size)), k=1)
        # Freeze the mask so it's not included in gradient computation
        # Equivalent to PyTorch's register_buffer
        self._no_grad.add("mask")

    def __call__(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        # Make the batch size and number of heads the batch dimension
        # so that attention computation can be done in parallel
        k = k.reshape(B, T, self.n_heads, self.head_size).transpose(0, 2, 1, 3)
        q = q.reshape(B, T, self.n_heads, self.head_size).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.head_size).transpose(0, 2, 1, 3)

        attn_scores = (q @ k.transpose(0, 1, 3, 2)) / mx.sqrt(
            mx.array(self.head_size, dtype=mx.float32)
        )
        # Use mask directly - mx.where handles float masks correctly
        attn_scores = mx.where(
            self.mask[:, :, :T, :T] > 0,
            mx.array(float("-inf"), dtype=mx.float32),
            attn_scores,
        )
        attn_weights = mx.softmax(attn_scores, axis=-1)

        attn_output = (
            attn_weights @ v
        )  # (B, n_heads, T, T) x (B, n_heads, T, head_size) -> (B, n_heads, T, head_size)

        # Re-assemble all head outputs side by side
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, T, C)

        output = self.c_proj(attn_output)
        return output


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)

    def __call__(self, x):
        x = self.c_fc(x)
        x = nn.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layers: int = 12
    n_heads: int = 12
    n_embed: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = {
            "wte": nn.Embedding(config.vocab_size, config.n_embed),
            "wpe": nn.Embedding(config.block_size, config.n_embed),
            "h": [Block(config) for _ in range(config.n_layers)],
            "ln_f": nn.LayerNorm(config.n_embed),
        }

    def __call__(self, idx, training=False):
        # The shape of idx is B, T
        # Because we are passing in a batch of sequences, where B is the batch size and T is the sequence length.
        # And each sequence is represented as a 1D array of token indices.
        B, T = idx.shape
        assert T <= self.config.block_size, (
            "Cannot forward, model block size is exhausted."
        )

        # Token and position embeddings
        token_embeddings = self.transformer["wte"](idx)  # (B, T, n_embed)
        position_ids = mx.arange(0, T, dtype=mx.int32).reshape(1, T)
        position_embeddings = self.transformer["wpe"](position_ids)  # (1, T, n_embed)

        x = token_embeddings + position_embeddings  # (B, T, n_embed)

        # Transformer blocks
        for block in self.transformer["h"]:
            x = block(x)

        x = self.transformer["ln_f"](x)  # (B, T, n_embed)

        # Weight tying: use embedding weights for output projection
        # This is equivalent to: logits = x @ self.transformer["wte"].weight.T
        logits = self.transformer["wte"].as_linear(x)  # (B, T, vocab_size)
        return logits
