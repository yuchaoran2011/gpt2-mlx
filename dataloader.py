import os

import mlx.core as mx
import numpy as np


def load_tokens(filename):
    npt = np.load(filename)
    return mx.array(npt)


class DataLoaderLite:
    def __init__(self, B, T, split="train"):
        self.B = B
        self.T = T
        assert split in ("train", "val")

        # Uncomment below if reading from a single file named input.txt
        # with open("input.txt", "r") as f:
        #     text = f.read()
        # enc = tiktoken.get_encoding("gpt2")
        # input_ids = enc.encode(text)
        # self.tokens = mx.array(input_ids)

        # Assume fineweb edu dataset is stored in a local directory edu_fineweb10B
        data_root = "edu_fineweb10B"
        shards = os.listdir(os.path.join(os.path.dirname(__file__), data_root))
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no data shards found for split {split}"

        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches of size {B}")
        self.current_position = 0

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        # Always read until B*T+1 so that the last token has its target
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).reshape(B, T)  # inputs
        y = (buf[1:]).reshape(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            print(f"loaded {len(self.tokens)} tokens from shard {self.current_shard}")
            self.current_position = 0
        return x, y
