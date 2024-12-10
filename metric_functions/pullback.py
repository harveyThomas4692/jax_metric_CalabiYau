import jax
import jax.numpy as jnp
from jax import jit

jit_setdiff1d = jit(jnp.setdiff1d, static_argnames=['size'])

def pullback(pt, projective_factors, poly):
    """
    Computes the pullback matrix for a given point, and polynomial.
    Args:
        pt (array-like): The point at which to evaluate the pullback.
        projective_factors (list): A list of integers representing the projective factors of the Calabi-Yau manifold.
        poly (function): The polynomial function for which the gradient is computed.
    Returns:
        jnp.ndarray: The pullback matrix.
    """
    
    dPoly = jax.grad(poly,holomorphic=True)
    dP = dPoly(pt)
    scales = len(projective_factors)
    pt_ones = jnp.argsort(jnp.abs(pt))[-scales:]
    args = jnp.argsort(jnp.abs(dP))

    # Delete the specified rows
    rm_index = jit_setdiff1d(args, pt_ones,size=1)[0]


    chain = jnp.outer(dP, 1./dP)  # This has indices pb_b^a = dp^a / dp_b

    # Update the desired column with zeros and ones
    pb = jnp.eye(len(pt),dtype=pt.dtype)
    pb = pb.at[:,pt_ones].set(jnp.zeros((len(pt),len(pt_ones))))
    pb = pb.at[:,rm_index].set(chain[:,rm_index])


    keep_indices = jit_setdiff1d(jnp.arange(len(pt)),jnp.append(pt_ones, rm_index),size=len(pt)-1-len(pt_ones))
    pb = pb[keep_indices, :]

    return jnp.array(pb)

pullback = jit(pullback,static_argnums=(2,))