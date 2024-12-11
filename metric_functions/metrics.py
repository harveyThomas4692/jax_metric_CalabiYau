import jax
import jax.numpy as jnp
from jax import jit, vmap, grad

@jit
def get2FormFSAmb(pt):
    '''
    Compute the FS two form in the ambient space, at a point - ASSUMES 1 PROJECTIVE SPACE
    '''
    n = len(pt)
    ptConj = jnp.conjugate(pt)

    g = (jnp.identity(n)*(jnp.linalg.norm(pt)**2.) - jnp.outer(ptConj,pt))/(jnp.linalg.norm(pt)**4.)

    return g