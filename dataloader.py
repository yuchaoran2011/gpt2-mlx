import mlx.core as mx
import tiktoken


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open("input.txt", "r") as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        input_ids = enc.encode(text)
        self.tokens = mx.array(input_ids)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches of size {B}")
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
            self.current_position = 0
        return x, y
