import logging
import os

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

logger = logging.getLogger("mlx-gpt")


def save_checkpoint(model, optimizer, step, checkpoint_dir, is_final=False):
    """Save model weights, optimizer state, and step number."""
    checkpoint_name = "final" if is_final else f"checkpoint_step_{step}"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    os.makedirs(checkpoint_path, exist_ok=True)

    # Save model weights
    model_weights_path = os.path.join(checkpoint_path, "model_weights.safetensors")
    model.save_weights(model_weights_path)

    # Save optimizer state (flatten nested structure)
    optimizer_state_path = os.path.join(checkpoint_path, "optimizer_state.npz")
    flat_state_list = tree_flatten(optimizer.state)
    flat_state = dict(flat_state_list)
    mx.savez(optimizer_state_path, **flat_state)

    # Save step number and other metadata
    metadata_path = os.path.join(checkpoint_path, "metadata.npz")
    mx.savez(metadata_path, step=mx.array(step))

    logger.info(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_dir):
    """Load the latest checkpoint if it exists. Returns starting step."""
    # Look for checkpoints
    if not os.path.exists(checkpoint_dir):
        return 0

    # Find all checkpoint directories
    checkpoints = []
    for item in os.listdir(checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(checkpoint_path):
            metadata_path = os.path.join(checkpoint_path, "metadata.npz")
            if os.path.exists(metadata_path):
                metadata = mx.load(metadata_path)
                step = int(metadata["step"].item())
                checkpoints.append((step, checkpoint_path))

    if not checkpoints:
        return 0

    # Load the latest checkpoint (highest step number)
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    step, checkpoint_path = checkpoints[0]

    # Load model weights
    model_weights_path = os.path.join(checkpoint_path, "model_weights.safetensors")
    if os.path.exists(model_weights_path):
        model.load_weights(model_weights_path)
        mx.eval(model.parameters())

    # Load optimizer state (unflatten nested structure)
    optimizer_state_path = os.path.join(checkpoint_path, "optimizer_state.npz")
    if os.path.exists(optimizer_state_path):
        flat_state = mx.load(optimizer_state_path)
        optimizer_state = tree_unflatten(flat_state)
        optimizer.state = optimizer_state

    logger.info(f"Resumed training from checkpoint at step {step} ({checkpoint_path})")
    return step + 1  # Return next step to start from
