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
    """Samples latents and creates a grid of generated samples

    Args:
        params (Any): Model params. Usually a frozendict
        model (flax.linen.module): Flax linen module
        rng (_type_): rng key for sampling latents
        num_samples (int, optional): Number of samples to generate Defaults to 64.

    Returns:
        plt.figure: Grid of generated samples
    """

    latents = random.normal(rng, (num_samples, model.num_latents))

    def generate(model):
        return model.generate(latents)

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
    """Convert numpy array of samples to matplotlib fig

    Args:
        array (np.array): Batch of original or reconstructed samples
        num_samples (int, optional): Number of samples to plot. Defaults to 16.

    Returns:
        plt.figure: Grid of batch
    """

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


def latents_to_scatter(latents: np.array, labels: np.array) -> plt.figure:
    """Plots scatter plot of first 2 latent variables

    Args:
        latents (np.array): Array of latent means
        labels (_type_): Array of labels

    Returns:
        plt.figure: Scatterplot of latents
    """

    fig = plt.figure()

    scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels)
    plt.xlabel(r"$\mu_z(x)_0$")
    plt.ylabel(r"$\mu_z(x)_1$")
    plt.title("Latent posterior means")
    plt.legend(handles=scatter.legend_elements()[0], labels=['0','1','2','3','4','5','6','7','8','9'])

    return fig
