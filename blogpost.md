# Learnings from Implementing GPT-2 from Scratch using MLX

This blog post documents my experience implementing a GPT-2 transformer model from scratch using Apple's MLX framework. Inspired by Andrej Karpathy's [NanoGPT](https://github.com/karpathy/nanoGPT), what started as a fun exercise to deepen my understanding of transformer architectures and the MLX framework turned into a rewarding journey filled with surprising learnings and challenges.

## Motivation

Transformers have revolutionized natural language processing, and GPT-2 is a seminal model in this space. Even though it's now considered ancient in the fast-evolving LLM space, implementing it from scratch still provides rich educational value, allowing me to explore the intricacies of transformer architectures, attention mechanisms, and model training. Additionally, using MLX provided an opportunity to optimize the model for Apple Silicon, leveraging its unified memory architecture.

---
**NOTE**

[MLX Transform LM example](https://github.com/ml-explore/mlx-examples/blob/main/transformer_lm/main.py) provided by the official examples repo was helpful as I built out my implementation.

---

## Key Learnings

I'll skip the basics of transformers and GPT-2 architecture, assuming familiarity. Here are some specific learnings from the implementation process:

### 1. Understanding MLX's Unified Memory Model

Apple Silicon offers a unified memory architecture that allows CPU and GPU to share the same memory space. This simplifies data management and reduces overhead from data transfers. Common pitfalls in PyTorch such as ensuring required tensors are on the right device are automatically handled. However, it also requires careful consideration of memory access patterns to avoid bottlenecks that may dramatically reduce training token throughput.

For more details on MLX's unified memory approach, see the [MLX documentation](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html).

### 2. MLX's Lazy Evaluation: A Paradigm Shift from PyTorch

The most significant difference between MLX and PyTorch is **lazy evaluation**. Understanding this design philosophy is crucial for writing efficient MLX code.

#### How It Works

In PyTorch, operations execute eagerly (e.g., when you write `z = x + y`, the addition happens immediately). In MLX, operations build a computation graph that only executes when explicitly materialized via `mx.eval()` or when the result is needed (e.g., converting to a Python scalar or NumPy array).

This is similar to JAX's approach but distinct from PyTorch's eager execution model. The MLX documentation provides a [detailed explanation of lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html).

**Advantages:**

- **Performance optimization**: MLX can optimize the entire computation graph by fusing operations
- **Simplified code**: No need for context managers like `torch.no_grad()` or switching between `model.train()` and `model.eval()` modes
- **Better pipelining**: Operations can be queued and executed asynchronously, maximizing hardware utilization

**Challenges:**
- **Memory management**: Without explicit materialization, computation graphs can grow unbounded, causing the program to crash
- **Debugging complexity**: Errors may not surface until evaluation time, making debugging less intuitive. Code that involves timing a code block may not produce accurate duration numbers if tensors are not materialized until later
- **Performance pitfalls**: Unnecessary `mx.eval()` calls can disrupt the optimization pipeline and significantly lower performance

#### Practical Implications

##### 1. Cannot Call `mx.eval()` Inside Compiled Functions

MLX's `@mx.compile` decorator optimizes functions by analyzing and transforming computation graphs. Since `mx.eval()` triggers immediate execution, it cannot be called within compiled functions—doing so would break the compilation process. I had to structure my code so evaluation happens outside compiled boundaries. In my `train.py`, only the forward and backward pass through the network are included in the compiled function for this reason.

##### 2. Gradient Accumulation Requires Periodic Evaluation

While `mx.eval()` isn't strictly required for correctness, I found it essential for gradient accumulation. 

When simulating large batch sizes through gradient accumulation (hundreds or thousands of steps), MLX builds the entire computation graph in memory. Without periodic `mx.eval()` calls to materialize intermediate results, the graph grows unbounded. On my M3 Max, this led to memory exhaustion and laptop crashes.

```python
# Without periodic evaluation - memory usage can quickly exceed local Mac's memory capacity
for i in range(num_accumulation_steps):
    loss = forward_and_backward(batch)
    # Graph keeps growing...

# With periodic evaluation - memory stays bounded
for i in range(num_accumulation_steps):
    loss = forward_and_backward(batch)
    if (i + 1) % eval_frequency == 0:
        mx.eval(loss)  # Materialize and free graph memory
```

##### 3. No Need for `torch.no_grad()` or `model.eval()`

In PyTorch, you must explicitly disable gradient tracking with `torch.no_grad()` during inference and switch models between training and evaluation modes:

```python
# PyTorch inference
model.eval()
with torch.no_grad():
    output = model(input)
```

MLX simplifies this significantly. Since gradients are only computed when you explicitly call gradient functions (like `*mx.value_and_grad(loss_fn)()`), there's no need for context managers:

```python
# MLX inference - just call the model
output = model(input)
```

This makes code cleaner and reduces the cognitive overhead of managing model states.

##### 4. The Learning Rate Logging Trap

I encountered a subtle performance issue while logging learning rates during training. My learning rate schedule combined linear warmup with cosine decay:

```python
linear_schedule = optim.linear_schedule(max_lr / warmup_steps, max_lr, warmup_steps)
cosine_schedule = optim.cosine_decay(max_lr, max_steps - warmup_steps, min_lr)
lr_schedule = optim.join_schedules([linear_schedule, cosine_schedule], [warmup_steps])
```

The natural way to log the current learning rate is:

```python
current_lr = mx.eval(lr_schedule(step))
```

However, this **noticeably degraded performance**—training throughput dropped from 11,500 tokens/sec to 7,500 tokens/sec on my M3 Max, a 35% decrease!

The issue: calling `mx.eval()` at every training step disrupted MLX's ability to pipeline and optimize the computation graph. The solution was to implement a separate pure Python function that mathematically calculates the cosine decay value (`get_lr_python` in `train.py`).

### 3. Mixed-Precision Training: Currently Missing

One notable limitation is that MLX currently doesn't support mixed-precision training (using float16/bfloat16 for computation while maintaining float32 for certain operations that require a higher precision). This is a standard technique in PyTorch. Mixed-precision training allows for taking advantage of the performance gains of bfloat16 but not sacrifice model quality for calculations that need them including softmax and gradient accumulation. I opened a [GitHub issue](https://github.com/ml-explore/mlx/issues/2834) to track this feature request.

### 4. Training Loop: MLX vs PyTorch

The training loop structure differs significantly between MLX and PyTorch, reflecting their different design philosophies.

#### PyTorch Training Loop

```python
model.train()

for epoch in range(num_epochs):
    for batch in dataloader:
      	optimizer.zero_grad()
        # Move data to device
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
```

#### MLX Training Loop

```python
# Define loss function and create gradient function
def loss_fn(model, inputs, targets):
    logits = model(inputs)
    return mx.mean(nn.losses.cross_entropy(logits, targets))

# Create gradient function
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, targets = batch
        # No need to move to device - unified memory
        
        # Forward and backward in one call
        loss, grads = loss_and_grad_fn(model, inputs, targets)
        
        # Optimizer step
        optimizer.update(model, grads)
        
        # Materialize loss for logging
        # Casting the loss to scalar float(loss) also triggers materialization
        mx.eval(loss)
```

#### Key Differences

1. **No explicit device management**: Unified memory eliminates `.to(device)` calls
2. **Functional gradient computation**: `nn.value_and_grad()` returns both loss and gradients in one call, rather than imperatively calling `.backward()`
3. **No `zero_grad()` needed**: Each gradient computation is independent; there's no persistent gradient state to clear

### 5. Vocabulary Size Padding

To optimize performance, I rounded GPT-2's vocabulary size (50,257) up to 50,304, a power of 2, as suggested by Karpathy. This is a common optimization that improves memory alignment and computational efficiency.

However, this created a subtle bug during inference. The model's output layer now produces logits for 50,304 tokens, including 47 invalid tokens that don't exist in the actual vocabulary. During sampling, these invalid tokens could be selected, causing inference failures.

The solution is to mask invalid token logits before sampling:

```python
def generate(model, tokens, max_new_tokens, temperature=1.0):
    VOCAB_SIZE = 50257  # Actual GPT-2 vocab size
    MODEL_VOCAB_SIZE = 50304  # Padded size
    
    for _ in range(max_new_tokens):
        logits = model(tokens)
        logits = logits[:, -1, :]  # Get last token logits
        
        # Mask invalid tokens by setting their logits to -inf
        logits[:, VOCAB_SIZE:] = float('-inf')
        
        # Apply temperature and sample
        logits = logits / temperature
        probs = mx.softmax(logits, axis=-1)
        next_token = mx.random.categorical(probs)
        
        tokens = mx.concatenate([tokens, next_token[:, None]], axis=1)
    
    return tokens
```

**The lesson**: Architectural optimizations can have downstream effects. Always validate that inference respects the actual vocabulary boundaries.

## Conclusion

Implementing GPT-2 from scratch using MLX was an enlightening experience. The framework's lazy evaluation model requires a different mental model than PyTorch's eager execution, but it offers powerful optimization opportunities once you understand the paradigm. The unified memory architecture of Apple Silicon, combined with MLX's design philosophy, creates a compelling platform for ML development on Mac.

While MLX is still maturing, it's already a powerful tool for both learning and practical ML development on Apple Silicon. I hope these learnings help you navigate your own MLX journey!
