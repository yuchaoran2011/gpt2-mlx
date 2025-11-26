import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.optimizers.optimizers import clip_grad_norm

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

optimizer = optim.AdamW(learning_rate=lr_schedule, betas=[0.9, 0.95], eps=1e-8)
optimizer.init(model.trainable_parameters())

optimizer_decay = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)
optimizer_skip_decay = optim.AdamW(learning_rate=1e-3, weight_decay=0.0)


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
state = [model.state, optimizer.state]


@partial(mx.compile, inputs=state, outputs=state)
def step(x, y):
    # When calling value_and_grad(), it computes the gradients from scratch for that specific forward pass.
    # There's no accumulation happening in the background.
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad_fn(model, x, y)

    # Implement gradient clipping and weight decay
    grads_to_decay, grads_to_skip_decay = {}, {}
    split_grads_by_weight_decay(grads, grads_to_decay, grads_to_skip_decay)

    clipped_grads_to_decay, _ = clip_grad_norm(grads_to_decay, max_norm=1.0)
    clipped_grads_to_skip_decay, _ = clip_grad_norm(grads_to_skip_decay, max_norm=1.0)
    optimizer_decay.update(model, clipped_grads_to_decay)
    optimizer_skip_decay.update(model, clipped_grads_to_skip_decay)
    return loss


for i in range(50):
    # Run 'sudo asitop' to monitor CPU usage
    t0 = time.time()
    x, y = train_loader.next_batch()
    # Unlike in PyTorch, no need for zero_grad() in MLX.
    loss = step(x, y)

    # Evaluate state after update to ensure changes are applied (MLX is lazy)
    mx.eval(state)
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (x.shape[0] * x.shape[1]) / (t1 - t0)

    # Evaluate loss to get actual value
    loss_val = float(loss)
    print(f"step {i}: loss {loss_val}, dt {dt:.2f}ms, tok/sec {tokens_per_sec:.2f}")

output = generate_text(
    model,
    prompt="Once upon a time",
    max_new_tokens=50,
    temperature=1,
    top_k=40,
)
print(output)
