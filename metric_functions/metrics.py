import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
from metric_functions.pullback import get_pullback
from metric_functions.complex_numbers import manual_det_3x3,grad_del_delBar

def get_2form_FS_proj(n,pts):
    """
    Compute the Fubini-Study 2-form for a set of points.
    This function calculates the Fubini-Study 2-form for each point in the input array `pts`.
    The Fubini-Study metric is a Hermitian metric on complex projective space.
    Args:
        n (int): The dimension of the complex projective space.
        pts (array-like): An array of points in complex projective space. Each point is expected to be a complex vector of length `n+1`.
    Returns:
        jax.numpy.ndarray: An array of Fubini-Study 2-forms corresponding to each input point.
    """

    g = vmap(lambda pt: (jnp.identity(n+1)*(jnp.linalg.norm(pt)**2.) - jnp.outer(jnp.conjugate(pt),pt))/(jnp.linalg.norm(pt)**4.))(pts)

    return g

get_2form_FS_proj = jit(get_2form_FS_proj, static_argnums=(0,))


def get_2form_FS_proj_prod(projective_factors, k_moduli, pts):
    """
    Computes the 2-form Fubini-Study direct product metric for given projective factors, moduli, and points.
    Args:
        projective_factors (tuple of int): List of projective factors for each block.
        k_moduli (list of float): List of moduli corresponding to each projective factor.
        pts (array-like): Array of points at which the metric is evaluated.
    Returns:
        jnp.ndarray: The computed 2-form Fubini-Study projection product metric.
    """

    metric = jnp.zeros((len(pts),len(pts[0]),len(pts[0])))*(1.0+0.0j)
    min = 0
    for i in range(len(projective_factors)):
        factor = projective_factors[i]
        block = get_2form_FS_proj(projective_factors[i],pts)
        metric = k_moduli[i]*metric.at[:,min:min+factor+1, min:min+factor+1].set(block)
        min += factor +1

    return metric

get_2form_FS_proj_prod = jit(get_2form_FS_proj_prod, static_argnums=(0,))

def get_ref_metric(projective_factors,k_moduli, poly, pts):
    """
    Compute the reference metric for a given set of points.
    This function calculates the reference metric by first obtaining the pullbacks
    and the Fubini-Study metrics, and then contracting them using Einstein summation.
    Parameters:
        projective_facotrs (tuple): The projective factors.
        k_moduli (array-like): The Kähler moduli.
        poly (array-like): The polynomial coefficients.
        pts (array-like): The points at which to evaluate the metric.
    Returns:
        jnp.ndarray: The computed reference metric.
    """
    
    pullbacks = get_pullback(pts,projective_factors,poly)
    fs_metrics = get_2form_FS_proj_prod(projective_factors,k_moduli,pts)

    return jnp.einsum('aij,ajk,alk->ail',pullbacks,fs_metrics,jnp.conjugate(pullbacks))

def __cy_vol_form_point(projective_factors,poly,pt):
    """
    Compute the volume form at a given point on a Calabi-Yau manifold.
    Parameters:
    projective_factors (tuple): Factors related to the projective coordinates.
    poly (function): A polynomial function representing the Calabi-Yau manifold.
    pt (array-like): A point in the manifold where the volume form is evaluated.
    Returns:
    float: The volume form at the given point.
    """

    dPoly = jax.grad(poly,holomorphic=True)
    dP = dPoly(pt)
    scales = len(projective_factors)
    pt_ones = jnp.argsort(jnp.abs(pt))[-scales:]
    dP_compare = jnp.abs(dP).at[pt_ones].set(jnp.zeros(len(pt_ones)))
    # Delete the specified rows
    rm_index = jnp.argsort(jnp.abs(dP_compare))[-1]
    dpRed = dP[rm_index]
    Womg = 1./jnp.abs(dpRed)**2.
    return Womg

__cy_vol_form_point = jit(__cy_vol_form_point,static_argnums=(0,1))

def cy_vol_form(projective_factors,poly,pts):
    """
    Computes the Calabi-Yau volume form for a set of points.
    Args:
        projective_factors (tuple): The projective factors used in the computation.
        poly (array-like): The polynomial coefficients defining the Calabi-Yau manifold.
        pts (array-like): A collection of points at which to evaluate the volume form.
    Returns:s
        array-like: The computed volume form at each point in `pts`.
    """
    
    return vmap(lambda pt: __cy_vol_form_point(projective_factors,poly,pt))(pts)

cy_vol_form = jit(cy_vol_form,static_argnums=(0,1))

def aux_weight(projective_factors,k_moduli, poly, pts):
    """
    Compute auxiliary weights for given projective factors, moduli, polynomial, and points.
    This function calculates the auxiliary weights by first obtaining the reference metric
    using the provided projective factors, moduli, polynomial, and points. It then computes
    the determinant of the 3x3 matrices of the reference metric.
    Args:
        projective_factors (tuple): The projective factors used in the computation.
        k_moduli (array-like): The moduli parameters.
        poly (array-like): The polynomial coefficients or terms.
        pts (array-like): The points at which the metric is evaluated.
    Returns:
        array-like: The computed auxiliary weights.
    """

    gCY = get_ref_metric(projective_factors,k_moduli,poly,pts)
    Wfs = jnp.abs(vmap(manual_det_3x3)(gCY))
    return Wfs

aux_weight = jax.jit(aux_weight, static_argnums=(0,2,))

def mass(projective_factors,k_moduli, poly, pts):
    """
    Calculate the masses for numer integration given the projective factors, 
    Kähler moduli, polynomial, and points.
    Args:
        projective_factors (tuple): The projective factors of the manifold.
        k_moduli (array-like): The Kähler moduli of the manifold.
        poly (array-like): The polynomial defining the manifold.
        pts (array-like): The points at which to evaluate the mass.
    Returns:
        float: The mass of the Calabi-Yau manifold at the given points.
    """

    Wfs = aux_weight(projective_factors,k_moduli,poly,pts)
    cy_vol_form_pts = cy_vol_form(projective_factors,poly,pts)
    return cy_vol_form_pts/Wfs

mass = jit(mass, static_argnums=(0,2,))

def normalised_mass(projective_factors,k_moduli, poly, pts):
    """
    Compute the normalized mass for given projective factors, Kähler moduli, polynomial, and points.
    This function calculates the masses using the provided projective factors, Kähler moduli, polynomial, 
    and points, then normalizes these masses by the total volume form of the Calabi-Yau manifold.
    Args:
        projective_factors (tuple): The projective factors used in the computation.
        k_moduli (array-like): The Kähler moduli used in the computation.
        poly (array-like): The polynomial defining the Calabi-Yau manifold.
        pts (array-like): The points at which the computation is performed.
    Returns:
        array-like: The normalized masses.
    Note:
        This is computing what are called auxiliary weights in the CY metric paper.
    """

    masses = mass(projective_factors,k_moduli,poly,pts)
    cy_vol = cy_vol_form(projective_factors,poly,pts)

    mTot = jnp.sum(masses)
    omgTot = jnp.sum(cy_vol)

    return masses * omgTot / mTot

normalised_mass = jit(normalised_mass, static_argnums=(0,2,))

def kappa(projective_factors,k_moduli, poly, pts):
    """
    Computes the kappa value for the Monge-Ampère (MA) loss.

    Parameters:
    projective_factors (tuple): The projective factors used in the computation.
    k_moduli (array-like): The Kähler moduli parameters.
    poly (array-like): The polynomial coefficients.
    pts (array-like): The points at which the computation is performed.

    Returns:
    float: The computed kappa value, which is the ratio of the sum of auxiliary weights to the sum of the Calabi-Yau volume form.
    """
    cy_vol = cy_vol_form(projective_factors,poly,pts)
    Wfs = aux_weight(projective_factors,k_moduli,poly,pts)

    return jnp.sum(Wfs)/jnp.sum(cy_vol)

kappa = jit(kappa, static_argnums=(0,2,))