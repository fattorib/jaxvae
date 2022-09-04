"""
Pseudocode for train loop


sample batch of data x 

split rng to get rng_key 

pass x and rng_key through model 

output here are logits of regenerated image (also have to pass out the generated latent )

two separate loss terms:

1. loss to match x vs p(x|z) (optax.sigmoid_binary_cross_entropy(logits, labels))
2. loss to match prior and generated latent 
+ 2.1 logprobs from normal around (0,0)
- 2.2 logprobs from normal around (mu,var)

"""

from optax import sigmoid_binary_cross_entropy
import jax.numpy as jnp 

def lognormal_pdf(sample, mean, logvar):
    log2pi = jnp.log(2. * jnp .pi)
    return jnp.sum(
      -.5 * ((sample - mean) ** 2. * jnp.exp(-logvar) + logvar + log2pi),
      axis=1)


def vae_loss(logits, x, z, mu, var):
    px_z_loss = sigmoid_binary_cross_entropy(logits,x)

    pz_loss = lognormal_pdf(z,0,0)

    qz_x_loss = lognormal_pdf(z,mu,var)

    return -(px_z_loss + pz_loss - qz_x_loss).mean()



