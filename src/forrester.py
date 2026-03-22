import jax.numpy as jnp

def forrester_funct(x):
    """
    The Forrester test function.
    """
    return (6.0 * x - 2.0)**2 * jnp.sin(12.0 * x - 4.0)
