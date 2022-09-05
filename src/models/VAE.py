from typing import Any
from jax import numpy as jnp
from flax import linen as nn
import jax

ModuleDef = Any
dtypedef = Any


class Decoder(nn.Module):
    @nn.compact
    def __call__(self, z, rng_key):
        mu, logvar = jnp.split(z, indices_or_sections=2, axis=-1)
        eps = jax.random.normal(key=rng_key, shape=logvar.shape)
        z = mu + eps * jnp.exp(0.5 * logvar)
        # decoder: learning $p_\theta (x|z)$ where z is a latent and x is a generated sample
        z = nn.Dense(features=7 * 7 * 32)(z)
        z = z.reshape(-1, 7, 7, 32)
        z = nn.relu(z)
        z = nn.ConvTranspose(
            kernel_size=(3, 3), strides=(2, 2), features=64, padding="SAME"
        )(z)
        z = nn.relu(z)
        z = nn.ConvTranspose(
            kernel_size=(3, 3), strides=(2, 2), features=32, padding="SAME"
        )(z)
        z = nn.relu(z)

        out = nn.ConvTranspose(
            kernel_size=(3, 3), strides=(1, 1), features=1, padding="SAME"
        )(z)

        return (out, mu, logvar)


class Encoder(nn.Module):
    num_latents: int = 5

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(kernel_size=(3, 3), strides=(2, 2), features=32)(x)
        x = nn.relu(x)
        x = nn.Conv(kernel_size=(3, 3), strides=(2, 2), features=64)(x)
        x = nn.relu(x)
        x = jnp.mean(x, axis=(1, 2))
        z = nn.Dense(features=2 * self.num_latents)(x)
        return z



class VAE(nn.Module):

    """Convolutional Variational Autoencoder as described in `Convolutional Variational Autoencoder`
    <https://www.tensorflow.org/tutorials/generative/cvae>
    """

    num_latents: int = 5

    def setup(self):

        self.encoder = Encoder(self.num_latents)
        self.decoder = Decoder()

    def __call__(self, x, rng_key):
        z = self.encoder(x)
        return self.decoder(z, rng_key)

    def generate(self, z, rng_key):
        return nn.sigmoid(self.decoder(z, rng_key)[0])
