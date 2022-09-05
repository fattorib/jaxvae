from typing import Any
from jax import numpy as jnp
from flax import linen as nn
import jax

ModuleDef = Any
dtypedef = Any

def reparametrize(z, rng_key):
    # reparamatrization trick: used to ensure we can backprop through mean & variance
    mu, logvar = jnp.split(z, indices_or_sections=2, axis=-1)
    eps = jax.random.normal(key=rng_key, shape=logvar.shape)
    z = mu + eps * jnp.exp(0.5 * logvar)

    return z, (mu, logvar)


class Decoder(nn.Module):
    @nn.compact
    def __call__(self, z):

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

        return out 


class Encoder(nn.Module):
    num_latents: int

    @nn.compact
    def __call__(self, x):

        # encoder: learning $q_\phi (z|x)$ where z is the generated latent and x is a datapoint
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

    num_latents: int

    def setup(self):

        self.encoder = Encoder(self.num_latents)
        self.decoder = Decoder()

    def __call__(self, x, rng_key):
        z = self.encoder(x)

        z, (mu,logvar) = reparametrize(z, rng_key)

        return (self.decoder(z), mu,logvar)

    def generate(self, z):
        return nn.sigmoid(self.decoder(z))
