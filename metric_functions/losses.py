from jax import jit, vmap
import jax.numpy as jnp

from metric_functions.complex_numbers import manual_det_3x3
from metric_functions.metrics import cy_vol_form, cy_metric, normalised_mass

def loss_ma(model, params,projective_factors,k_moduli, poly,kappa_val, pts):
    """
    Computes the Monge-Ampere loss for a given model.
    This function calculates the loss by comparing the determinant of the 
    Calabi-Yau metric to the CY volume form.
    Parameters:
    model (callable): The model used to compute the metric.
    params (dict): Parameters for the model.
    projective_factors (tuple): Projective factors for the computation.
    k_moduli (array-like): Moduli parameters for the computation.
    poly (array-like): Polynomial coefficients.
    kappa_val (array-like): Kappa value for the computation. Computed using kappa on the whole dataset.
    pts (array-like): Points at which to evaluate the metric.
    Returns:
    jnp.ndarray: The mean absolute loss value.
    """

    #det = vmap(manual_det_3x3)(cy_metric(model, params,projective_factors,k_moduli, poly, pts))
    det = vmap(jnp.linalg.det)(cy_metric(model, params,projective_factors,k_moduli, poly, pts))

    omg = cy_vol_form(projective_factors,poly,pts)
    
    return jnp.mean(jnp.abs(1.- (det/(kappa_val*omg))))

loss_ma = jit(loss_ma,static_argnums=(0,2,4,))

def sigma_measure(model, params,projective_factors,k_moduli, poly,kappa_val, pts):
    """
    Computes the sigma measure for a given model.
    This function calculates the sigma measure by comparing the determinant of the 
    Calabi-Yau metric to the product of the volume form.
    Parameters:
    model (callable): The model used to compute the metric.
    params (dict): Parameters for the model.
    projective_factors (tuple): Projective factors for the computation.
    k_moduli (array-like): Moduli parameters for the computation.
    poly (array-like): Polynomial coefficients.
    kappa_val (array-like): Kappa value for the computation. Computed using kappa on the whole dataset.
    pts (array-like): Points at which to evaluate the metric.
    Returns:
    jnp.ndarray: The mean absolute loss value.
    """
    
    #det = vmap(manual_det_3x3)(cy_metric(model, params,projective_factors,k_moduli, poly, pts))
    det = vmap(jnp.linalg.det)(cy_metric(model, params,projective_factors,k_moduli, poly, pts))

    omg = cy_vol_form(projective_factors,poly,pts)
    normalised_weights = normalised_mass(projective_factors, k_moduli, poly, pts)
    
    return jnp.sum(normalised_weights*jnp.abs(1.- ((det)/(kappa_val*omg))))


sigma_measure = jit(sigma_measure,static_argnums=(0,2,4,))