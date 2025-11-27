import gc
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.optimizers.optimizers import clip_grad_norm
from mlx.utils import tree_map

from dataloader import DataLoaderLite
from gpt2 import GPT, GPTConfig
from inference import generate_text

model = GPT(GPTConfig(vocab_size=50304))
# Initialize model parameters (MLX is lazy by default)
mx.eval(model.parameters())

# Similar to autocast in PyTorch, we convert the model to bfloat16 for faster training
model.apply(lambda x: x.astype(mx.bfloat16))

train_loader = DataLoaderLite(B=16, T=1024)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
linear_schedule = optim.linear_schedule(max_lr * 1 / warmup_steps, max_lr, warmup_steps)
cosine_schedule = optim.cosine_decay(max_lr, max_steps - warmup_steps, min_lr)
lr_schedule = optim.join_schedules([linear_schedule, cosine_schedule], [warmup_steps])

optimizer_decay = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)
optimizer_skip_decay = optim.AdamW(learning_rate=1e-3, weight_decay=0.0)

total_batch_size = 524288  # 2**19, ~0.5M number of tokens
B = 16  # micro-batch size
T = 1024  # sequence length
assert total_batch_size % (B * T) == 0, (
    "make sure total_batch_size is divisible by B * T"
)
grad_accum_steps = total_batch_size // (B * T)
print(f"Using gradient accumulation over {grad_accum_steps} steps")


def split_grads_by_weight_decay(grads, grads_to_decay, grads_to_skip_decay):
    for k, v in grads.items():
        if isinstance(v, dict):
            # Need to preserve the nested dictionary structure
            # Otherwise two parameters with the same name under different modules would collide
            # e.g. "layer1.weight" vs "layer2.weight"
            grads_to_decay[k] = {}
            grads_to_skip_decay[k] = {}
            split_grads_by_weight_decay(v, grads_to_decay[k], grads_to_skip_decay[k])

            # Remove empty nested dicts to avoid confusing the optimizer
            if not grads_to_decay[k]:
                del grads_to_decay[k]
            if not grads_to_skip_decay[k]:
                del grads_to_skip_decay[k]
        # Deal with a list of Block modules in the hidden layers
        elif isinstance(v, list):
            grads_to_decay[k] = []
            grads_to_skip_decay[k] = []
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    decay_dict = {}
                    skip_dict = {}
                    split_grads_by_weight_decay(item, decay_dict, skip_dict)
                    grads_to_decay[k].append(decay_dict)
                    grads_to_skip_decay[k].append(skip_dict)
                else:
                    if item.ndim < 2:
                        grads_to_skip_decay[k].append(item)
                    else:
                        grads_to_decay[k].append(item)
        else:
            if v.ndim < 2:
                grads_to_skip_decay[k] = v
            else:
                grads_to_decay[k] = v


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
state = [model.state, optimizer_decay.state, optimizer_skip_decay.state]


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

    # Implement gradient clipping and weight decay
    # https://github.com/ml-explore/mlx/issues/2837 has more details
    grads_to_decay, grads_to_skip_decay = {}, {}
    split_grads_by_weight_decay(grads_accum, grads_to_decay, grads_to_skip_decay)

    clipped_grads_to_decay, _ = clip_grad_norm(grads_to_decay, max_norm=1.0)
    clipped_grads_to_skip_decay, _ = clip_grad_norm(grads_to_skip_decay, max_norm=1.0)
    optimizer_decay.update(model, clipped_grads_to_decay)
    optimizer_skip_decay.update(model, clipped_grads_to_skip_decay)
    mx.eval(model.parameters())
    return loss_accum


for i in range(3):
    # Run 'sudo asitop' to monitor CPU usage
    t0 = time.time()

    # Unlike in PyTorch, no need for zero_grad() in MLX.
    loss = step()
    # Evaluate model parameters after compiled function returns
    # This ensures optimizer updates are materialized and helps with memory management
    mx.eval(model.parameters())
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps) / (t1 - t0)

    # loss is already a Python float from step(), but ensure it's not an array
    loss_val = float(loss) if isinstance(loss, mx.array) else loss
    print(f"step {i}: loss {loss_val}, dt {dt:.2f}ms, tok/sec {tokens_per_sec:.2f}")

output = generate_text(
    model,
    prompt="Once upon a time",
    max_new_tokens=50,
    temperature=1,
    top_k=40,
)
print(output)
