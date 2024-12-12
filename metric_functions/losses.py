from jax import jit, vmap
import jax.numpy as jnp

from metric_functions.complex_numbers import manual_det_3x3
from metric_functions.metrics import cy_vol_form, cy_metric, kappa

def loss_ma(model, params,projective_factors,k_moduli, poly, pts):
    '''
        Computes the MA loss at a point
    '''
    det = vmap(manual_det_3x3)(cy_metric(model, params,projective_factors,k_moduli, poly, pts))

    omg = cy_vol_form(projective_factors,poly,pts)
    kappa_val = kappa(projective_factors,k_moduli, poly, pts)

    return jnp.mean(jnp.abs(1.- (det/(kappa_val*omg))))

loss_ma = jit(loss_ma,static_argnums=(0,2,4,))