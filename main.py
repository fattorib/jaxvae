import numpy as np
import optax
import functools

from jax import random

# Flax imports
from flax.training import train_state
import jax.numpy as jnp
import jax

# PyTorch - for dataloading
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from src.utils.dataloader import * 

from src.training.train_utils import *

# Logging/Config Stuffs
import argparse
import logging 
from omegaconf import OmegaConf
from aim import Run

logging.basicConfig(level=logging.INFO)


def parse():
    parser = argparse.ArgumentParser(description="VAE Training")

    parser.add_argument("--cfg", default="conf/config.yaml", type=str)

    args = parser.parse_args()
    return args


def main():
    args = parse()
    cfg = OmegaConf.load(args.cfg)

    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            FlattenAndCast(),
        ]
    )

    train_dataset = MNIST(
        root=f"data/MNIST",
        train=True,
        download=True,
        transform=transform_train,
    )

    train_loader = NumpyLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.workers,
        pin_memory=False,
    )

    run = Run()

    run["hparams"] = {
        "learning_rate": cfg.training.lr,
        "batch_size": cfg.training.batch_size,
        "latent_dimension": cfg.model.latent_dim,
    }

    # --------- Create Train State ---------#
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    model = XYZ 
    learning_rate_fn = XYZ

    state = create_train_state(
        init_rng,
        learning_rate_fn=learning_rate_fn,
        weight_decay=args.weight_decay,
        model=model,
    )

def initialized(key, image_size, model):
    input_shape = (1, image_size, image_size, 1)

    @jax.jit
    def init(rng, shape):
        return model.init(rng, shape)

    variables = init(rng=key, shape=jnp.ones(input_shape))
    return variables["params"]


def create_train_state(rng, learning_rate_fn, weight_decay, model):
    """Creates initial `TrainState`."""
    params = initialized(rng, 28, model)
    mask = jax.tree_map(lambda x: x.ndim != 1, params)
    tx = optax.adamw(learning_rate=learning_rate_fn, weight_decay=weight_decay)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )
    return state


@jax.jit
def train_step(state, batch, rng_key, labels):
    """Train for a single step."""

    def loss_fn(params):
        out, new_state = state.apply_fn(
            {"params": params},
            batch,
            rng_key,
            None

        )

        logits, extra = out 
        z, mu, var = extra 
        loss = vae_loss(logits=logits, x = batch, z=z, mu= mu, var=var)


        return loss, (logits, new_state)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    loss, grads = grad_fn(state.params)

    state = state.apply_gradients(
        grads=grads,
    )

    # NEED TO LOG PARAMS TOO
    metrics = {"VAE Loss": loss}

    return state, metrics


if __name__ == '__main__':
    main()