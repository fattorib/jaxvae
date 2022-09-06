import numpy as np
from jax import random

# Flax imports
import jax
import optax

# PyTorch - for dataloading
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST
from src.utils.dataloader import *

# Model imports
from src.models.VAE import VAE

# Utils
from src.training.losses import *
from src.utils.sample import sample_from_latents, np_to_fig, latents_to_scatter
from src.training.training_utils import create_train_state

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
    validation_dataset = MNIST(
        root=f"data/MNIST",
        train=False,
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
        "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
    }

    # --------- Create Train State ---------#
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    model = VAE(num_latents=cfg.model.latent_dim)

    if cfg.training.use_schedule:
        # Warmup over 1 epoch, decay LR to 0 over remaining epochs
        steps_per_epoch = (
            len(train_loader) // cfg.training.gradient_accumulation_steps
        )
        learning_rate_fn = optax.warmup_cosine_decay_schedule(
            init_value=0,
            peak_value=cfg.training.learning_rate,
            warmup_steps=steps_per_epoch,
            decay_steps=(cfg.training.epochs - 1) * steps_per_epoch,
        )

    else:
        learning_rate_fn = (cfg.training.learning_rate,)

    state = create_train_state(
        init_rng,
        learning_rate_fn,
        weight_decay=cfg.training.weight_decay,
        model=model,
        grad_accum_steps=cfg.training.gradient_accumulation_steps,
    )

    del init_rng

    for epoch in tqdm(range(0, cfg.training.epochs)):
        rng, subrng = jax.random.split(rng)
        state, train_metrics_np = train_epoch(state, subrng, train_loader)

        (
            validation_metrics,
            original,
            reconstructed,
            (latent_array, labels_array),
        ) = eval_epoch(state, subrng, validation_loader)

        labels_array = np.concatenate(labels_array).reshape(
            -1,
        )
        latent_array = np.concatenate(latent_array).reshape(
            -1, cfg.model.latent_dim
        )

        generated_grid = sample_from_latents(
            state.params["params"],
            VAE(num_latents=cfg.model.latent_dim),
            subrng,
        )

        # Logging to Aim
        original, reconstructed = np_to_fig(
            jax.device_get(original).reshape(-1, 28, 28)
        ), np_to_fig(jax.device_get(reconstructed).reshape(-1, 28, 28))

        latent_fig = latents_to_scatter(latent_array, labels_array)

        aim_image = Image(
            generated_grid, format="png", optimize=True, quality=90
        )

        aim_original = Image(original, format="png", optimize=True, quality=90)

        aim_reconstructed = Image(
            reconstructed, format="png", optimize=True, quality=90
        )

        aim_scatter = Image(latent_fig, format="png", optimize=True, quality=90)

        run.track(
            aim_image,
            name="Latent Generations",
            step=epoch,
            context={"subset": "train"},
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
            aim_scatter,
            name="Latent Scatter",
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
        prior_loss = kl_divergence(mean, logvar)

        return loss, prior_loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, prior_loss), grads = grad_fn(state.params)

    state = state.apply_gradients(
        grads=grads,
    )

    metrics = {
        "VAE Loss": loss,
        "Prior Loss": prior_loss,
        "Reconstruction Loss": loss - prior_loss,
    }

    return state, metrics


@jax.jit
def eval_step(state, batch, rng_key):
    """Validate a single batch."""

    mean_array = []

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

    mean_array.append(jax.device_get(mean))

    return metrics, batch, logits, mean_array


def eval_epoch(state, rng, dataloader):
    """Validation loop"""
    batch_metrics = []
    latent_array = []
    labels_array = []

    for i, (batch, labels) in tqdm(
        enumerate(dataloader), total=len(dataloader)
    ):
        rng, subrng = random.split(rng)
        metrics, original, reconstructed, batch_latents = eval_step(
            state,
            batch,
            subrng,
        )
        batch_metrics.append(metrics)
        latent_array.append(batch_latents)
        labels_array.append(labels)

    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }

    return (
        epoch_metrics_np,
        original,
        reconstructed,
        (latent_array, labels_array),
    )


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
