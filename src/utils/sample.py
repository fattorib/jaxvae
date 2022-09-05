""" 
Helper code for sampling from VAE
"""
import jax
from jax import random
import flax
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from typing import Any


def sample_from_latents(
    params: Any, model: flax.linen.module, rng, num_samples: int = 64
) -> plt.figure:

    latents = random.normal(rng, (num_samples, 2 * model.num_latents))

    def generate(model):
        return model.generate(latents, rng)

    generated = flax.linen.apply(generate, model)({"params": params})

    np_images = jax.device_get(generated).reshape(num_samples, 28, 28)

    fig = plt.figure()

    ncols, nrows = int(jnp.sqrt(num_samples)), int(jnp.sqrt(num_samples))

    axes = [
        fig.add_subplot(nrows, ncols, r * ncols + c + 1)
        for r in range(0, nrows)
        for c in range(0, ncols)
    ]

    i = 0
    for ax in axes:
        ax.imshow(np_images[i, :], cmap="gray")
        i += 1

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    return fig


def np_to_fig(array: np.array, num_samples: int = 16) -> plt.figure:
    """Convert numpy array of samples to matplotlib fig"""

    fig = plt.figure()

    ncols, nrows = int(jnp.sqrt(num_samples)), int(jnp.sqrt(num_samples))

    axes = [
        fig.add_subplot(nrows, ncols, r * ncols + c + 1)
        for r in range(0, nrows)
        for c in range(0, ncols)
    ]

    i = 0

    for ax in axes:
        ax.imshow(
            np.where(array[i, :] > 0.5, 1.0, 0.0).astype("float32"), cmap="gray"
        )
        i += 1

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    return fig
