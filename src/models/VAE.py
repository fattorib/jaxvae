from typing import Any
from jax import numpy as jnp
from flax import linen as nn

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
    def __call__(self, x, z = None):
        
        # encoder: learning $q_\phi (z|x)$ where x is input image and z is the latent
        if z is not None:
            x = nn.Conv(kernel_size=(3,3), strides=(2,2), features = 32)(x)
            x = nn.relu(x)
            x = nn.Conv(kernel_size=(3,3), strides=(2,2), features = 64)(x)
            x = nn.relu(x)
            x = jnp.mean(x, axis=(1, 2))
            z = nn.Dense(features= 2 * self.num_latents)

        # decoder: learning $p_\theta (x|z)$ where z is a latent and x is a generated sample
        z = nn.Dense(features = 7*7*32)(z)
        z = z.reshape(7,7,32)
        z = nn.ConvTranspose(kernel_size=(3,3), strides=(2,2), features = 64, padding = 'SAME')(z)
        z = nn.relu(z)
        z = nn.ConvTranspose(kernel_size=(3,3), strides=(2,2), features = 32, padding = 'SAME')(z)
        z = nn.relu(z)

        z = nn.ConvTranspose(kernel_size=(3,3), strides=(1,1), features = 1, padding = 'SAME')(z)

        return z 








