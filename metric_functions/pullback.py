import jax
import jax.numpy as jnp
from jax import jit

jit_setdiff1d = jit(jnp.setdiff1d, static_argnames=['size'])

def pullback(pt, poly):
    """
    Computes the pullback matrix for a given point, and polynomial.
    Args:
        pt (array-like): The point at which to evaluate the pullback.
        poly (function): The polynomial function for which the gradient is computed.
    Returns:
        jnp.ndarray: The pullback matrix.
    """
    
    dPoly = jax.grad(poly,holomorphic=True)
    dP = dPoly(pt)
    pt_ones = jnp.where(jnp.real(pt) == 1.0)[0]
    args = jnp.argsort(jnp.abs(dP))

    # Delete the specified rows
    rm_index = jnp.setdiff1d(args, pt_ones)[0]


    chain = jnp.outer(dP, 1./dP)  # This has indices pb_b^a = dp^a / dp_b

    # Update the desired column with zeros and ones
    pb = jnp.eye(len(pt))
    pb = pb.at[:,pt_ones].set(jnp.zeros((len(pt),len(pt_ones))))
    pb = pb.at[:,rm_index].set(chain[:,rm_index])


    keep_indices = jnp.setdiff1d(jnp.arange(len(pt)),jnp.append(pt_ones, rm_index))
    pb = pb[keep_indices, :]

    return pb

#pullback = jit(pullback,static_argnums=(2,))