""" 
Helper methods used during training setup. 
"""
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state


def initialized(key, image_size, model):
    """Initializes param dict for a model

    Args:
        key (_type_): _description_
        image_size (_type_): _description_
        model (_type_): _description_

    Returns:
        _type_: _description_
    """
    input_shape = (1, image_size)

    rng_init, rng_model = jax.random.split(key)

    @jax.jit
    def init(rng, shape, key):
        return model.init(rng, shape, key)

    variables = init(rng=rng_init, shape=jnp.ones(input_shape), key=rng_model)
    return variables


def create_train_state(
    rng, learning_rate_fn, weight_decay, model, grad_accum_steps
):
    """Creates initial `TrainState` for model."""
    params = initialized(rng, 784, model)
    mask = jax.tree_map(lambda x: x.ndim != 1, params)

    tx = optax.adamw(
        learning_rate=learning_rate_fn, weight_decay=weight_decay, mask=mask
    )

    if grad_accum_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=grad_accum_steps)

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )
    return state
