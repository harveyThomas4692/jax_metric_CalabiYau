import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
from metric_functions.pullback import get_pullback

def get_2form_FS_proj(n,pts):
    """
    Compute the Fubini-Study projection 2-form for a set of points.
    This function calculates the Fubini-Study projection 2-form for each point in the input array `pts`.
    The Fubini-Study metric is a Hermitian metric on complex projective space.
    Args:
        n (int): The dimension of the complex projective space.
        pts (array-like): An array of points in complex projective space. Each point is expected to be a complex vector of length `n+1`.
    Returns:
        jax.numpy.ndarray: An array of Fubini-Study projection 2-forms corresponding to each input point.
    """

    g = vmap(lambda pt: (jnp.identity(n+1)*(jnp.linalg.norm(pt)**2.) - jnp.outer(jnp.conjugate(pt),pt))/(jnp.linalg.norm(pt)**4.))(pts)

    return g

get_2form_FS_proj = jit(get_2form_FS_proj, static_argnums=(0,))


def get_2form_FS_proj_prod(projective_factors, k_moduli, pts):
    """
    Computes the 2-form Fubini-Study projection product metric for given projective factors, moduli, and points.
    Args:
        projective_factors (list of int): List of projective factors for each block.
        k_moduli (list of float): List of moduli corresponding to each projective factor.
        pts (array-like): Array of points at which the metric is evaluated.
    Returns:
        jnp.ndarray: The computed 2-form Fubini-Study projection product metric.
    Note:
        This function is currently incompatible with JAX's `jit` compilation.
    """

    metric = jnp.zeros((len(pts),len(pts[0]),len(pts[0])))*(1.0+0.0j)
    min = 0
    for i in range(len(projective_factors)):
        factor = projective_factors[i]
        block = get_2form_FS_proj(projective_factors[i],pts)
        metric = k_moduli[i]*metric.at[:,min:min+factor+1, min:min+factor+1].set(block)
        min += factor +1

    return metric

def get_ref_metric(projective_facotrs,k_moduli, poly, pts):
    """
    Compute the reference metric for a given set of points.
    This function calculates the reference metric by first obtaining the pullbacks
    and the Fubini-Study metrics, and then contracting them using Einstein summation.
    Parameters:
    projective_facotrs (array-like): The projective factors.
    k_moduli (array-like): The Kähler moduli.
    poly (array-like): The polynomial coefficients.
    pts (array-like): The points at which to evaluate the metric.
    Returns:
    jnp.ndarray: The computed reference metric.
    Note: This function cannot be JIT-compiled due to the use of `get_2form_FS_proj_prod`.
    """
    
    pullbacks = get_pullback(pts,projective_facotrs,poly)
    fs_metrics = get_2form_FS_proj_prod(projective_facotrs,k_moduli,pts)

    return jnp.einsum('ijk,ikl,ibk->ijb',pullbacks,fs_metrics,pullbacks.conj())
