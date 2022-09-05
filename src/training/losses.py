import jax.numpy as jnp
import jax
import flax.linen as nn

@jax.vmap
def kl_divergence(mean: jnp.array, logvar: jnp.array) -> jnp.array:
    """KL Divergence for standard normal prior

    Args:
        mean (jnp.array): Array of latent means
        logvar (jnp.array): Array of latent logvars

    Returns:
        jnp.array: KL Divergence
    """
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


@jax.jit
def vae_loss(logits: jnp.array, x: jnp.array, mean: jnp.array, logvar:jnp.array) -> jnp.array:
    """VAE loss function. Sum of 
        -'reconstruction' loss
        -'prior-matching' loss

    Args:
        logits (jnp.array): Output logits from model. These correspond to reconstructed batch from VAE
        x (jnp.array): Batch ground truth
        mean (jnp.array): Array of latent means
        logvar (jnp.array): Array of latent logvars

    Returns:
        jnp.array: VAE loss
    """
    logits = nn.log_sigmoid(logits)
    px_z_loss = -jnp.sum(
        x * logits + (1.0 - x) * jnp.log(-jnp.expm1(logits))
    ).mean()

    kl_loss = kl_divergence(mean, logvar).mean()

    return px_z_loss + kl_loss
