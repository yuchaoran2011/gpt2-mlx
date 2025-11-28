import mlx.core as mx
import tiktoken

from train import VOCAB_SIZE


def generate_text(model, prompt, max_new_tokens=100, temperature=1.0, top_k=None):
    enc = tiktoken.get_encoding("gpt2")
    input_ids = enc.encode(prompt)
    input_ids = mx.array(input_ids, dtype=mx.int32).reshape(1, -1)  # Batch size of 1

    while input_ids.shape[1] < max_new_tokens + len(enc.encode(prompt)):
        logits = model(input_ids)  # Forward pass through the model
        logits = logits[:, -1, :] / temperature  # Focus on the last token

        # Ensure logits shape matches VOCAB_SIZE
        logits = logits[:, :VOCAB_SIZE]

        if top_k is not None:
            # top_k_values shape: (batch_size, top_k)
            top_k_values = mx.topk(logits, k=top_k, axis=-1)
            # min_top_k is the smallest value in the top_k_values
            min_top_k = top_k_values[:, -1].reshape(-1, 1)
            logits = mx.where(logits < min_top_k, mx.array(float("-inf")), logits)

        probs = mx.softmax(logits, axis=-1)  # Convert logits to probabilities
        next_token = mx.random.categorical(
            probs, num_samples=1
        )  # Sample the next token

        input_ids = mx.concat([input_ids, next_token], axis=1)  # Append to input_ids

    output_ids = input_ids[0].tolist()  # Convert to list
    output_text = enc.decode(output_ids)  # Decode to text
    return output_text
