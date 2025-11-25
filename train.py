import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from dataloader import DataLoaderLite
from gpt2 import GPT, GPTConfig
from inference import generate_text

model = GPT(GPTConfig())
# Initialize model parameters (MLX is lazy by default)
mx.eval(model.parameters())

# Similar to autocast in PyTorch, we convert the model to bfloat16 for faster training
model.apply(lambda x: x.astype(mx.bfloat16))

train_loader = DataLoaderLite(B=16, T=1024)

optimizer = optim.AdamW(learning_rate=1e-4)
optimizer.init(model.trainable_parameters())


# When deriving gradients, MLX tracks gradients for all trainable parameters automatically
# Training parameters don't need to be an argument for loss_fn
def loss_fn(x, y):
    y = y.reshape(-1)
    # Shift logits and targets to align them for next-token prediction
    logits = model(x)
    logits = logits.reshape(-1, model.config.vocab_size)

    # The dimensions of loss will be (B*T,), we take the mean to get a single scalar loss value
    # Note that this behavior differ from that of F.cross_entropy in PyTorch where it automatically returns the mean loss
    return nn.losses.cross_entropy(logits, y, reduction="mean")


for i in range(10):
    # Run 'sudo asitop' to monitor CPU usage
    t0 = time.time()
    x, y = train_loader.next_batch()
    # Unlike in PyTorch, no need for zero_grad() in MLX.
    # When calling value_and_grad(), it computes the gradients from scratch for that specific forward pass.
    # There's no accumulation happening in the background.
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad_fn(x, y)
    optimizer.update(model, grads)
    # Evaluate parameters after update to ensure changes are applied (MLX is lazy)
    mx.eval(model.parameters())
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
