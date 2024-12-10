import jax
import jax.numpy as jnp
from jax import jit

jit_setdiff1d = jit(jnp.setdiff1d, static_argnames=['size'])

def pullback(pt, projective_factors, poly):
    '''
    Get the pullback map at a point on a CY manifold.
    '''
    pb = 1
    return pb

pullback = jit(pullback,static_argnums=(2,))