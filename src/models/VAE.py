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
        z = nn.Dense(features=7 * 7 * 32)(z)
        z = z.reshape(-1, 7, 7, 32)
        z = nn.relu(z)
        z = nn.ConvTranspose(
            kernel_size=(3, 3), strides=(2, 2), features=64, padding="SAME", kernel_init=self.kernel_init, bias_init=self.bias_init
        )(z)
        z = nn.relu(z)
        z = nn.ConvTranspose(
            kernel_size=(3, 3), strides=(2, 2), features=32, padding="SAME", kernel_init=self.kernel_init, bias_init=self.bias_init
        )(z)
        z = nn.relu(z)

        out = nn.ConvTranspose(
            kernel_size=(3, 3), strides=(1, 1), features=1, padding="SAME", kernel_init=self.kernel_init, bias_init=self.bias_init
        )(z)

        return out


class Encoder(nn.Module):
    num_latents: int
    kernel_init: Callable = nn.initializers.normal(stddev=0.01,)
    bias_init: Callable = nn.initializers.normal(stddev=0.01,)

    @nn.compact
    def __call__(self, x):

        # encoder: learning $q_\phi (z|x)$ where z is the generated latent and x is a datapoint
        x = nn.Conv(kernel_size=(3, 3), strides=(2, 2), features=32, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        x = nn.relu(x)
        x = nn.Conv(kernel_size=(3, 3), strides=(2, 2), features=64, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        x = nn.relu(x)
        x = jnp.mean(x, axis=(1, 2))
        z_mean = nn.Dense(features=self.num_latents, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        z_logvar = nn.Dense(features=self.num_latents, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        return z_mean, z_logvar


# class Decoder(nn.Module):

#     kernel_init: Callable = nn.initializers.normal(stddev=0.01,)
#     bias_init: Callable = nn.initializers.normal(stddev=0.01,)
    
#     @nn.compact
#     def __call__(self, z):
#         # decoder: learning $p_\theta (x|z)$ where z is a latent and x is a generated sample
#         z = nn.Dense(features=500, name="fc1", kernel_init=self.kernel_init, bias_init=self.bias_init)(z)
#         z = nn.relu(z)
#         out = nn.Dense(784, name="fc2", kernel_init=self.kernel_init, bias_init=self.bias_init)(z)

#         return out


# class Encoder(nn.Module):
#     num_latents: int = 5

#     kernel_init: Callable = nn.initializers.normal(stddev=0.01,)
#     bias_init: Callable = nn.initializers.normal(stddev=0.01,)

#     @nn.compact
#     def __call__(self, x):
#         x = nn.Dense(500, name="fc1", kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
#         x = nn.relu(x)

#         # NOTE: Couldn't we just initialize them to what we expect? Ex: z_mean init is 0 kernel, 0 bias and z_logvar is 0 kernel, 1 bias?
#         z_mean = nn.Dense(features=self.num_latents, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
#         z_logvar = nn.Dense(features=self.num_latents, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
#         return z_mean, z_logvar


class VAE(nn.Module):

    """Convolutional Variational Autoencoder as described in `Convolutional Variational Autoencoder`
    <https://www.tensorflow.org/tutorials/generative/cvae>
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
