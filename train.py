import logging
import math
import os
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import tiktoken
from mlx.optimizers.optimizers import clip_grad_norm
from mlx.utils import tree_map

from checkpoint import load_checkpoint, save_checkpoint
from dataloader import DataLoaderLite
from gpt2 import GPT, GPTConfig
from inference import VOCAB_SIZE, generate_text

logging.basicConfig(
    filename=os.path.join(os.path.dirname(__file__), "logs.txt"),
    filemode="a",
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
# Get a logger instance
logger = logging.getLogger("mlx-gpt")

# Checkpoint directory
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
model = GPT(GPTConfig(vocab_size=VOCAB_SIZE))
# Initialize model parameters (MLX is lazy by default)
mx.eval(model.parameters())

enc = tiktoken.get_encoding("gpt2")

# Similar to autocast in PyTorch, we convert the model to bfloat16 for faster training
model.apply(lambda x: x.astype(mx.bfloat16))

train_loader = DataLoaderLite(B=16, T=1024, split="train")
val_loader = DataLoaderLite(B=16, T=1024, split="val")

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073
linear_schedule = optim.linear_schedule(max_lr * 1 / warmup_steps, max_lr, warmup_steps)
cosine_schedule = optim.cosine_decay(max_lr, max_steps - warmup_steps, min_lr)
lr_schedule = optim.join_schedules([linear_schedule, cosine_schedule], [warmup_steps])


def get_lr_python(step: int) -> float:
    """
    Compute learning rate using pure Python math (faster than MLX for logging).
    The alternative is to call mx.eval(lr_schedule(step)), which is much slower, causing token throughput
    to drop from ~11.5k to ~7.5k on M3 Max.
    """
    if step < warmup_steps:
        # Linear warmup
        init_lr = max_lr * 1 / warmup_steps
        return init_lr + step * ((max_lr - init_lr) / warmup_steps)
    else:
        # Cosine decay
        decay_steps = max_steps - warmup_steps
        step_in_decay = step - warmup_steps
        progress = min(step_in_decay / decay_steps, 1.0)
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + decay * (max_lr - min_lr)


# Create optimizers for parameters with and without weight decay
optimizer_decay = optim.AdamW(
    learning_rate=lr_schedule, weight_decay=0.01, betas=[0.9, 0.95]
)
optimizer_skip_decay = optim.AdamW(
    learning_rate=lr_schedule, weight_decay=0.0, betas=[0.9, 0.95]
)


# Filter function for MultiOptimizer: returns True for parameters that should have weight decay
# Parameters with ndim >= 2 (weights) get weight decay, ndim < 2 (biases/1D params) don't
def should_apply_weight_decay(path, param):
    """Returns True if parameter should have weight decay (ndim >= 2)."""
    return param.ndim >= 2


# Use MultiOptimizer to automatically route parameters to the correct optimizer
optimizer = optim.MultiOptimizer(
    optimizers=[optimizer_decay, optimizer_skip_decay],
    filters=[should_apply_weight_decay],
)

# Try to load checkpoint and resume training
start_step, _ = load_checkpoint(
    model, optimizer, checkpoint_dir, data_loader=train_loader
)

total_batch_size = 524288  # 2**19, ~0.5M number of tokens
B = 32  # micro-batch size
T = 1024  # sequence length
assert total_batch_size % (B * T) == 0, (
    "make sure total_batch_size is divisible by B * T"
)
grad_accum_steps = total_batch_size // (B * T)
logger.info(f"Using gradient accumulation over {grad_accum_steps} steps")


# When deriving gradients, MLX tracks gradients for all trainable parameters automatically
# Training parameters don't need to be an argument for loss_fn
def loss_fn(model, x, y):
    y = y.reshape(-1)
    # Shift logits and targets to align them for next-token prediction
    logits = model(x)
    logits = logits.reshape(-1, model.config.vocab_size)

    # The dimensions of loss will be (B*T,), we take the mean to get a single scalar loss value
    # Note that this behavior differ from that of F.cross_entropy in PyTorch where it automatically returns the mean loss
    return nn.losses.cross_entropy(logits, y, reduction="mean")


# See https://github.com/ml-explore/mlx-examples/blob/main/transformer_lm/main.py for an example of using mx.compile with optimizers
# I observed marginal improvements on M3 Max to training efficiency. token/sec went from 10k to 11.5k
# GPU was already close to full utilization
state = [model.state, optimizer.state]


# Create loss_and_grad_fn once outside the loop to avoid recreating it
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)


# Compile only the forward/backward pass, not the gradient accumulation loop
# This allows me to use mx.eval() during accumulation to free computation graphs
# otherwise memory will be quickly exhausted causing the machine to crash
@partial(mx.compile, inputs=state, outputs=state)
def forward_backward(x, y):
    """Compiled forward and backward pass for a single micro-batch."""
    loss, grads = loss_and_grad_fn(model, x, y)
    return loss, grads


def step():
    """Step function with gradient accumulation. Not compiled to allow mx.eval() calls."""
    loss_accum = 0.0
    grads_accum = None

    for _ in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        loss, grads = forward_backward(x, y)

        # Evaluate loss and gradients immediately to materialize them and free computation graph
        # This is critical for memory management during gradient accumulation
        mx.eval(loss, grads)

        # Accumulate gradients
        if grads_accum is None:
            grads_accum = grads
        else:
            # Unlike in PyTorch, each forward pass computes the gradients from scratch for that specific forward pass
            # There's no accumulation happening, hence manual accumulation via tree_map is needed
            # tree_map applies the given function recursively to each leaf node in the tree structure
            grads_accum = tree_map(lambda a, b: a + b, grads_accum, grads)
            # Evaluate the accumulated result immediately to free computation graph
            mx.eval(grads_accum)

        # Convert loss to float (triggers evaluation automatically)
        loss_val = float(loss) / grad_accum_steps
        loss_accum += loss_val

    # Ensure accumulated gradients are materialized before optimizer step
    mx.eval(grads_accum)

    # Implement gradient clipping
    # https://github.com/ml-explore/mlx/issues/2837 has more details
    clipped_grads, grad_norm = clip_grad_norm(grads_accum, max_norm=1.0)

    # MultiOptimizer automatically routes parameters to the correct optimizer
    # based on the filter function (weight decay for ndim >= 2, no decay for ndim < 2)
    optimizer.update(model, clipped_grads)
    mx.eval(model.parameters())
    return loss_accum, grad_norm


for i in range(start_step, max_steps):
    # Run 'sudo asitop' to monitor CPU usage
    t0 = time.time()

    # Save checkpoint periodically (every 1000 steps) and at validation steps (every 100 steps)
    if i > 0 and (i % 1000 == 0 or (i % 100 == 0 and i % 1000 != 0)):
        save_checkpoint(model, optimizer, i, checkpoint_dir, data_loader=train_loader)

    if i > 0 and i % 100 == 0 or i == max_steps - 1:
        val_loader.reset()
        val_loss_accum = 0.0
        val_batches = 10
        for _ in range(val_batches):
            x_val, y_val = val_loader.next_batch()
            val_loss = loss_fn(model, x_val, y_val)
            mx.eval(val_loss)
            val_loss_accum += float(val_loss)
        val_loss_avg = val_loss_accum / val_batches
        logger.info(f"step {i}: validation loss {val_loss_avg}")

        num_return_sequences = 3
        tokens = enc.encode("On a remote island, ")
        tokens = mx.array(tokens, dtype=mx.int32)
        tokens = mx.tile(mx.expand_dims(tokens, axis=0), (num_return_sequences, 1))
        logger.info(
            generate_text(model, prompt=tokens, max_new_tokens=50, temperature=0.2)
        )

    # Unlike in PyTorch, no need for zero_grad() in MLX.
    loss, grad_norm = step()
    # Evaluate model parameters after compiled function returns
    # This ensures optimizer updates are materialized and helps with memory management
    mx.eval(model.parameters())
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps) / (t1 - t0)

    # loss is already a Python float from step(), but ensure it's not an array
    loss_val = float(loss) if isinstance(loss, mx.array) else loss
    current_lr = get_lr_python(i)
    grad_norm_val = float(grad_norm)

    logger.info(
        f"step {i}: loss {loss_val}, grad_norm {grad_norm_val:.6f}, lr {current_lr:.2e}, dt {dt:.2f}ms, tok/sec {tokens_per_sec:.2f}"
    )

tokens = enc.encode("Once upon a time")
tokens = mx.array(tokens, dtype=mx.int32)
output = generate_text(
    model,
    prompt=tokens,
    max_new_tokens=50,
    temperature=1,
    top_k=40,
)
logger.info(output)

# Save final checkpoint
save_checkpoint(
    model,
    optimizer,
    max_steps - 1,
    checkpoint_dir,
    is_final=True,
    data_loader=train_loader,
)
logger.info("Training completed. Final checkpoint saved.")
