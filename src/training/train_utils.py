import jax.numpy as jnp
import flax 

def create_cos_anneal_schedule(base_lr, min_lr, max_steps):
    def learning_rate_fn(step):
        cosine_decay = (0.5) * (1 + jnp.cos(jnp.pi * step / max_steps))
        decayed = (1 - min_lr) * cosine_decay + min_lr
        return base_lr * decayed

    return learning_rate_fn


def compute_weight_decay(params):
    """Given a pytree of params, compute the summed $L2$ norm of the params.
    
    NOTE: For our case with SGD, weight decay ~ L2 regularization. This won't always be the 
    case (ex: Adam vs. AdamW).
    """
    param_norm = 0

    weight_decay_params_filter = flax.traverse_util.ModelParamTraversal(
        lambda path, _: ("bias" not in path and "scale" not in path)
    )

    weight_decay_params = weight_decay_params_filter.iterate(params)

    for p in weight_decay_params:
        if p.ndim > 1:
            param_norm += jnp.sum(p ** 2)

    return param_norm