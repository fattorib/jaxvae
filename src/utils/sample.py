""" 
Helper code for sampling from VAE
"""
import jax 

def generate_from_latent(model, rng_key):
    latent = jax.random.normal(rng_key, shape=(1, 2*model.latent_dim))
    return model(z = latent)