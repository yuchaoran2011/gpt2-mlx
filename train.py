import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import tiktoken

from gpt2 import GPT, GPTConfig
from inference import generate_text

model = GPT(GPTConfig())
# Initialize model parameters (MLX is lazy by default)
mx.eval(model.parameters())

with open("input.txt", "r") as f:
    text = f.read()

enc = tiktoken.get_encoding("gpt2")
input_ids = enc.encode(text)

B, T = 4, 32
buf = mx.array(input_ids[: B * T + 1])
x = buf[:-1].reshape(B, T)
y = buf[1:].reshape(B, T)

optimizer = optim.AdamW(learning_rate=1e-4)
optimizer.init(model.trainable_parameters())


# When deriving gradients, MLX tracks gradients for all trainable parameters automatically
# Training parameters don't need to be an argument for loss_fn
def loss_fn(x, y):
    y = y.reshape(-1)
    # Shift logits and targets to align them for next-token prediction
    logits = model(x)
    print(f"Model output dtype: {logits.dtype}")
    logits = logits.reshape(-1, model.config.vocab_size)

    # The dimensions of loss will be (B*T,), we take the mean to get a single scalar loss value
    # Note that this behavior differ from that of F.cross_entropy in PyTorch where it automatically returns the mean loss
    return nn.losses.cross_entropy(logits, y, reduction="mean")


for i in range(100):
    # Unlike in PyTorch, no need for zero_grad() in MLX.
    # When calling value_and_grad(), it computes the gradients from scratch for that specific forward pass.
    # There's no accumulation happening in the background.
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad_fn(x, y)
    optimizer.update(model, grads)
    # Evaluate parameters after update to ensure changes are applied (MLX is lazy)
    mx.eval(model.parameters())
    # Evaluate loss to get actual value
    loss_val = float(loss)
    print(f"step {i}: loss {loss_val}")

output = generate_text(
    model,
    prompt="Once upon a time",
    max_new_tokens=50,
    temperature=1,
    top_k=40,
)
print(output)
