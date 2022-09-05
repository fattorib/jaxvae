from os import stat
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

# Model imports
from src.models.VAE import VAE

from src.training.train_utils import *
from src.utils.sample import sample_from_latents, np_to_fig

# Logging/Config Stuffs
import argparse
import logging
from omegaconf import OmegaConf
from aim import Run, Image
from tqdm import tqdm

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

    validation_dataset = MNIST(
        root=f"data/MNIST",
        train=False,
        download=True,
        transform=transform_train,
    )

    validation_loader = NumpyLoader(
        validation_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.workers,
        pin_memory=False,
        drop_last=True,
    )

    run = Run()

    run["hparams"] = {
        "learning_rate": cfg.training.learning_rate,
        "batch_size": cfg.training.batch_size,
        "latent_dimension": cfg.model.latent_dim,
    }

    # --------- Create Train State ---------#
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    model = VAE(num_latents=cfg.model.latent_dim)

    state = create_train_state(
        init_rng,
        learning_rate_fn=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        model=model,
    )

    del init_rng

    for epoch in tqdm(range(0, cfg.training.epochs)):
        rng, subrng = jax.random.split(rng)
        state, train_metrics_np = train_epoch(state, subrng, train_loader)

        validation_metrics, original, reconstructed = eval_epoch(
            state, subrng, validation_loader
        )

        generated_grid = sample_from_latents(
            state.params["params"], VAE(num_latents=cfg.model.latent_dim), rng
        )

        rng = subrng

        original, reconstructed = np_to_fig(
            jax.device_get(original).reshape(-1, 28, 28)
        ), np_to_fig(jax.device_get(reconstructed).reshape(-1, 28, 28))

        aim_image = Image(
            generated_grid, format="png", optimize=True, quality=90
        )

        aim_original = Image(original, format="png", optimize=True, quality=90)

        aim_reconstructed = Image(
            reconstructed, format="png", optimize=True, quality=90
        )

        run.track(
            aim_image, name="images", step=epoch, context={"subset": "train"}
        )
        run.track(
            aim_original,
            name="Original Images",
            step=epoch,
            context={"subset": "train"},
        )
        run.track(
            aim_reconstructed,
            name="Reconstructed Images",
            step=epoch,
            context={"subset": "train"},
        )

        run.track(
            train_metrics_np["VAE Loss"],
            name="loss",
            step=epoch,
            context={"subset": "train"},
        )
        run.track(
            train_metrics_np["Prior Loss"],
            name="prior loss",
            step=epoch,
            context={"subset": "train"},
        )
        run.track(
            train_metrics_np["Reconstruction Loss"],
            name="reconstruction loss",
            step=epoch,
            context={"subset": "train"},
        )

        run.track(
            validation_metrics["VAE Loss"],
            name="loss",
            step=epoch,
            context={"subset": "validation"},
        )
        run.track(
            validation_metrics["Prior Loss"],
            name="prior loss",
            step=epoch,
            context={"subset": "validation"},
        )
        run.track(
            validation_metrics["Reconstruction Loss"],
            name="reconstruction loss",
            step=epoch,
            context={"subset": "validation"},
        )


def initialized(key, image_size, model):
    input_shape = (1, image_size, image_size, 1)

    @jax.jit
    def init(rng, shape):
        return model.init(rng, shape, rng)

    variables = init(rng=key, shape=jnp.ones(input_shape))
    return variables


def create_train_state(rng, learning_rate_fn, weight_decay, model):
    """Creates initial `TrainState`."""
    params = initialized(rng, 28, model)
    mask = jax.tree_map(lambda x: x.ndim != 1, params)
    tx = optax.adamw(
        learning_rate=learning_rate_fn, weight_decay=weight_decay, mask=mask
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )
    return state


@jax.jit
def train_step(state, batch, rng_key):
    """Train for a single step."""

    def loss_fn(params):
        logits, mean, logvar = state.apply_fn(
            {"params": params["params"]},
            batch,
            rng_key,
        )
        loss = vae_loss(logits=logits, x=batch, mean=mean, logvar=logvar)

        return loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grads = grad_fn(state.params)

    state = state.apply_gradients(
        grads=grads,
    )

    # NEED TO LOG PARAMS TOO
    # TODO: Not a shitty way to log this

    _, mean, logvar = state.apply_fn(
        {"params": state.params["params"]},
        batch,
        rng_key,
    )

    prior_loss = kl_divergence(mean, logvar)

    metrics = {
        "VAE Loss": loss,
        "Prior Loss": prior_loss,
        "Reconstruction Loss": loss - prior_loss,
    }

    return state, metrics


@jax.jit
def eval_step(state, batch, rng_key):
    """Train for a single step."""

    logits, mean, logvar = state.apply_fn(
        {"params": state.params["params"]},
        batch,
        rng_key,
    )
    loss = vae_loss(logits=logits, x=batch, mean=mean, logvar=logvar)

    prior_loss = kl_divergence(mean, logvar)

    metrics = {
        "VAE Loss": loss,
        "Prior Loss": prior_loss,
        "Reconstruction Loss": loss - prior_loss,
    }

    return metrics, batch, logits


def eval_epoch(state, rng, dataloader):
    """Validation loop"""
    batch_metrics = []

    for i, (batch, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        new_rng, subrng = random.split(rng)
        metrics, original, reconstructed = eval_step(
            state,
            batch,
            subrng,
        )
        batch_metrics.append(metrics)
        rng = new_rng

    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }
    return epoch_metrics_np, original, reconstructed


def train_epoch(state, rng, dataloader):
    """Train for a single epoch."""
    batch_metrics = []

    for i, (batch, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        new_rng, subrng = random.split(rng)
        state, metrics = train_step(
            state,
            batch,
            subrng,
        )
        batch_metrics.append(metrics)
        rng = new_rng

    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }
    return state, epoch_metrics_np


if __name__ == "__main__":
    main()
