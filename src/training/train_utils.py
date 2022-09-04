import jax.numpy as jnp
import jax 
from optax import sigmoid_binary_cross_entropy


def create_cos_anneal_schedule(base_lr, min_lr, max_steps):
    def learning_rate_fn(step):
        cosine_decay = (0.5) * (1 + jnp.cos(jnp.pi * step / max_steps))
        decayed = (1 - min_lr) * cosine_decay + min_lr
        return base_lr * decayed

    return learning_rate_fn

@jax.jit
def lognormal_pdf(sample, mean, logvar):
    log2pi = jnp.log(2. * jnp .pi)
    return jnp.sum(
      -.5 * ((sample - mean) ** 2. * jnp.exp(-logvar) + logvar + log2pi),
      axis=1)

@jax.jit
def vae_loss(logits, x, z, mu, var):
    px_z_loss = sigmoid_binary_cross_entropy(logits,x)

    pz_loss = lognormal_pdf(z,0,0)

    qz_x_loss = lognormal_pdf(z,mu,var)

    return -(px_z_loss + pz_loss - qz_x_loss).mean()
