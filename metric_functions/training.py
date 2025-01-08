from jax import jit
import jax.numpy as jnp

from metric_functions.complex_numbers import complex_to_real

def apply_model(model, params, x):
    """
    Applies the given model to the input data after converting it from complex to real representation,
    and then converts the output back to complex representation.
    Args:
        model: The model to be applied. It should have an 'apply' method that takes parameters and input data.
        params: The parameters for the model.
        x: The input data in complex representation.
    Returns:
        The output of the model in complex representation.
    """

    y = complex_to_real(x)
    return model.apply(params, y)

apply_model = jit(apply_model,static_argnums=(0,))

def apply_model_real(model, params, x):
    """
    Applies the given model to the input data in real representation.
    Args:
        model: The model to be applied. It should have an 'apply' method that takes parameters and input data.
        params: The parameters for the model.
        x: The input data in real representation.
    Returns:
        The output of the model in real representation.
    """

    return jnp.array([model.apply(params, x),0.])

apply_model_real = jit(apply_model_real,static_argnums=(0,))