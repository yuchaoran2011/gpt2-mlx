import mlx.core as mx
import tiktoken

VOCAB_SIZE = 50304  # GPT-2 vocabulary size


def generate_text(model, prompt, max_new_tokens=100, temperature=1.0, top_k=None):
    enc = tiktoken.get_encoding("gpt2")
    input_ids = prompt.astype(mx.int32)
    initial_length = input_ids.shape[1]

    while input_ids.shape[1] < max_new_tokens + initial_length:
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

    # Decode all sequences in the batch
    output_texts = []
    for i in range(input_ids.shape[0]):
        output_ids = input_ids[i].tolist()  # Convert to list
        output_text = enc.decode(output_ids)  # Decode to text
        output_texts.append(output_text)
    
    # Return single string if batch size is 1, otherwise return formatted string
    if len(output_texts) == 1:
        return output_texts[0]
    else:
        return "\n".join(f"[{i+1}] {text}" for i, text in enumerate(output_texts))
