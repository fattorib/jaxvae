from typing import Any
from jax import numpy as jnp
from flax import linen as nn
import jax 

ModuleDef = Any
dtypedef = Any

class VAE(nn.Module):

    """Convolutional Variational Autoencoder as described in `Convolutional Variational Autoencoder`
        <https://www.tensorflow.org/tutorials/generative/cvae>
    """
    num_latents: int = 5

    #TODO: UNUSED
    dtype: dtypedef = jnp.float32
    
    # # define init for conv layers. Do we need this?
    # kernel_init: Callable = nn.initializers.normal(stddev=0.02, dtype=dtype)

    @nn.compact
    def __call__(self, x, rng_key, z = None):
        
        # encoder: learning $q_\phi (z|x)$ where x is input image and z is the latent
        if z is None:
            x = nn.Conv(kernel_size=(3,3), strides=(2,2), features = 32)(x)
            x = nn.relu(x)
            x = nn.Conv(kernel_size=(3,3), strides=(2,2), features = 64)(x)
            x = nn.relu(x)
            x = jnp.mean(x, axis=(1, 2))
            z = nn.Dense(features= 2 * self.num_latents)(x)

        if rng_key is not None:
            # split up z latent into $\mu(x), \sigma^2 (x)$
            mu, var = jnp.split(z, indices_or_sections=2, axis = -1)
            eps = jax.random.normal(key = rng_key, shape = var.shape)
            latent = mu + eps*var 

        # decoder: learning $p_\theta (x|z)$ where z is a latent and x is a generated sample
        z = nn.Dense(features = 7*7*32)(latent)
        z = z.reshape(-1,7,7,32)
        z = nn.ConvTranspose(kernel_size=(3,3), strides=(2,2), features = 64, padding = 'SAME')(z)
        z = nn.relu(z)
        z = nn.ConvTranspose(kernel_size=(3,3), strides=(2,2), features = 32, padding = 'SAME')(z)
        z = nn.relu(z)

        out = nn.ConvTranspose(kernel_size=(3,3), strides=(1,1), features = 1, padding = 'SAME')(z)

        return (out,latent, mu, var)








