from typing import Any, Callable
from jax import numpy as jnp
from flax import linen as nn
import jax

ModuleDef = Any
dtypedef = Any


def reparametrize(mean, logvar, rng_key):
    # reparamatrization trick: used to ensure we can backprop through mean & variance
    eps = jax.random.normal(key=rng_key, shape=logvar.shape)
    z = mean + (eps * jnp.exp(0.5 * logvar))

    return z, (mean, logvar)


class Decoder(nn.Module):

    kernel_init: Callable = nn.initializers.normal(stddev=0.01,)
    bias_init: Callable = nn.initializers.normal(stddev=0.01,)
    
    @nn.compact
    def __call__(self, z):
        # decoder: learning $p_\theta (x|z)$ where z is a latent and x is a generated sample
        z = nn.Dense(features=500, name="fc1", kernel_init=self.kernel_init, bias_init=self.bias_init)(z)
        z = nn.relu(z)
        out = nn.Dense(784, name="fc2", kernel_init=self.kernel_init, bias_init=self.bias_init)(z)

        return out


class Encoder(nn.Module):
    num_latents: int = 5

    kernel_init: Callable = nn.initializers.normal(stddev=0.01,)
    bias_init: Callable = nn.initializers.normal(stddev=0.01,)

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(500, name="fc1", kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        x = nn.relu(x)

        # NOTE: Couldn't we just initialize them to what we expect? 
        # Ex: z_mean init is 0 kernel, 0 bias and z_logvar is 0 kernel, 1 bias?
        z_mean = nn.Dense(features=self.num_latents, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        z_logvar = nn.Dense(features=self.num_latents, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        return z_mean, z_logvar


class VAE(nn.Module):

    """Variational Autoencoder as described in `Autoencoding Variational Bayes`
    <https://arxiv.org/abs/1312.6114>
    """

    num_latents: int

    def setup(self):

        self.encoder = Encoder(self.num_latents)
        self.decoder = Decoder()

    def __call__(self, x, rng_key):
        z_mean, z_logvar = self.encoder(x)

        z, (mu, logvar) = reparametrize(z_mean, z_logvar, rng_key)

        return (self.decoder(z), mu, logvar)

    def generate(self, z):
        return nn.sigmoid(self.decoder(z))

    def extract_latents(self, x):
        z_mean, z_logvar = self.encoder(x)
        return z_mean, z_logvar
