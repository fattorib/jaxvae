import jax.numpy as jnp
import jax 
from optax import sigmoid_binary_cross_entropy
from jax.scipy.stats import norm

def create_cos_anneal_schedule(base_lr, min_lr, max_steps):
    def learning_rate_fn(step):
        cosine_decay = (0.5) * (1 + jnp.cos(jnp.pi * step / max_steps))
        decayed = (1 - min_lr) * cosine_decay + min_lr
        return base_lr * decayed

    return learning_rate_fn


@jax.vmap
def kl_divergence(mean, logvar):
  return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

@jax.jit
def vae_loss(logits, x, mean, logvar):
    px_z_loss = sigmoid_binary_cross_entropy(logits,x)

    kl_loss = kl_divergence(mean, logvar)

    return (px_z_loss + kl_loss).mean()

