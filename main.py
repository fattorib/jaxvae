import numpy as np
import optax
import functools

from jax import random

# Flax imports
from typing import Any
import flax
from flax.training.checkpoints import save_checkpoint
from flax.training import train_state
import jax.numpy as jnp
import jax

# PyTorch - for dataloading
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

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



if __name__ == '__main__':
    main()