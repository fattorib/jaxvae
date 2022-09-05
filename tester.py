import numpy as np
import optax
import functools

from jax import random

# Flax imports
from flax.training import train_state
import jax.numpy as jnp
import jax

# PyTorch - for dataloading
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from src.utils.dataloader import *

# Model imports
from src.models.VAE import VAE

from src.training.train_utils import *

# Logging/Config Stuffs
import argparse
import logging
from omegaconf import OmegaConf
from aim import Run
from tqdm import tqdm
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

image_size = 28
input_shape = (1, image_size, image_size, 1)

rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)
model = VAE(num_latents=2)

params = model.init(init_rng, jnp.ones(input_shape), init_rng)

logits, mu, var = model.apply(
    {"params": params["params"]}, jnp.ones(input_shape), rng
)


def sample_latents(params, rng, num_samples=64):

    latents = random.normal(rng, (64, 2 * model.num_latents))

    def generate(model):
        return model.generate(latents, rng)

    generated = flax.linen.apply(generate, VAE(num_latents=2))(
        {"params": params}
    )

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


plot = sample_latents(params["params"], rng)
plt.show()
