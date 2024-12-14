import jax
import jax.numpy as jnp
from jax import jit

@jit
def complex_to_real(z):
    """
    Converts a complex number to a real-valued array.
    Parameters:
    z (complex): A complex number.
    Returns:
    jnp.array: A 1D array with the real and imaginary parts of the complex number.
    Example: complex_to_real(3+4j)
    >>> jnp.array([3., 4.])
    """
    return jnp.array([z.real,z.imag])

@jit
def real_to_complex(x):
    """
    Converts a real-valued array to a complex number.
    Parameters:
    x (jnp.array): A 1D array with two elements, where the first element is the real part and the second element is the imaginary part.
    Returns:
    complex: A complex number constructed from the real and imaginary parts.
    Example:
    >>> real_to_complex(jnp.array([3, 4]))
    (3+4j)
    """
    return x[0]+1.j*x[1]

def grad_del(func, z):
    """
    Computes the holomorphic derivative of a given function at a complex point.
    Parameters:
    func (callable): A function that takes a complex number and returns a complex number.
    z (complex): The complex point at which to compute the holomorphic derivative.
    Returns:
    complex: The holomorphic derivative of the function at the given complex point.
    
    Example:
    >>> func = lambda z: z**2
    >>> grad_del(func, 1+1j)
    (2+2j)
    """
    tempFunc_R = lambda Z,W: jnp.real(func(Z+1.j*W))
    tempFunc_I = lambda Z,W: jnp.imag(func(Z+1.j*W))
    x = jnp.real(z)
    y = jnp.imag(z)
    dF1 = jax.jacrev(tempFunc_R,argnums=(0))(x,y) + 1.j * jax.jacrev(tempFunc_I,argnums=(0))(x,y)
    dF2 = jax.jacrev(tempFunc_R,argnums=(1))(x,y) + 1.j * jax.jacrev(tempFunc_I,argnums=(1))(x,y)

    return (dF1 - 1.j*dF2)/2.

grad_del = jit(grad_del, static_argnums=(0,))

def grad_del_real(func,X):
    """
    Computes the gradient of a real-valued function with respect to the real part of a complex variable.
    Args:
        func (callable): A function that takes a complex variable and returns a real value.
        X (array-like): A 2-element array representing the real and imaginary parts of the complex variable.
    Returns:
        jnp.ndarray: A 2-element array containing the gradient of the function with respect to the real part of the complex variable.
    """
    
    #x,y = X, for z = x+iy
    dFx = jax.jacfwd(func)(X)

    return jnp.array([(dFx[0,0] + dFx[1,1]), -(dFx[0,1] - dFx[1,0])])/2.

#grad_del_real = jit(grad_del_real, static_argnums=(0,))

def grad_delBar(func, z):
    """
    Computes the gradient of a given complex-valued function with respect to the conjugate of a complex variable.
    Parameters:
    func (callable): A function that takes a complex number and returns a complex number.
    z (complex): A complex number at which the gradient is evaluated.
    Returns:
    complex: The gradient of the function with respect to the conjugate of the complex variable.
    Example:
    >>> def example_func(z):
    ...     return z**2 + 1.j * z
    >>> z = 1 + 2.j
    >>> grad_delBar(example_func, z)
    DeviceArray(1.-1.j, dtype=complex64)
    """    
    tempFunc_R = lambda Z,W: jnp.real(func(Z+1.j*W))
    tempFunc_I = lambda Z,W: jnp.imag(func(Z+1.j*W))
    x = jnp.real(z)
    y = jnp.imag(z)
    dF1 = jax.jacfwd(tempFunc_R,argnums=(0))(x,y) + 1.j * jax.jacfwd(tempFunc_I,argnums=(0))(x,y)
    dF2 = jax.jacfwd(tempFunc_R,argnums=(1))(x,y) + 1.j * jax.jacfwd(tempFunc_I,argnums=(1))(x,y)

    return (dF1 + 1.j*dF2)/2.

grad_delBar = jit(grad_delBar, static_argnums=(0,))

def grad_delBar_real(func,X):
    """
    Computes the gradient of the real part of the function with respect to the complex conjugate variable.
    Parameters:
    func (callable): The function for which the gradient is to be computed. It should take a single argument.
    X (array-like): The input array representing the complex variable z = x + iy, where X = [x, y].
    Returns:
    jnp.ndarray: An array containing the computed gradient components.
    """

    #x,y = X, for z = x+iy

    dFx = jax.jacfwd(func)(X)

    return jnp.array([(dFx[0,0] - dFx[1,1]), (dFx[0,1] + dFx[1,0])])/2.

grad_delBar_real = jit(grad_delBar_real, static_argnums=(0,))

def grad_del_delBar(func, z):
    """
    Computes the matrix delDelBar(func).
    Parameters:
    func (callable): A function that takes a complex number and returns a complex number.
    z (complex): A complex number at which the delDelBar is to be computed.
    Returns:
    numpy.ndarray: The matrix resulting from the delDelBar operation on the function at point z.
    
    Example:
    >>> func = lambda z: z**2 * conj(z)
    >>> gradDelDelBar(func, 3.)
    DeviceArray(6.+0.j, dtype=complex64)
    """

    delBarFunc = lambda pt: grad_delBar(func,pt)

    delDelBar = grad_del(delBarFunc, z)

    return delDelBar

grad_del_delBar = jit(grad_del_delBar, static_argnums=(0,))

def grad_del_delBar_real(func, X):
    """
    Computes the matrix delDelBar(func) for a real-valued function.
    Parameters:
    func (callable): The function for which the matrix is to be computed.
    X (array-like): The input array representing the complex variable z = x + iy, where X = [x, y].
    Returns:
    jnp.ndarray: The matrix resulting from the delDelBar operation on the function at point z.
    """
    delBarFunc = lambda pt: grad_delBar_real(func, pt)

    def localDel(func, pt):
        dFx = jax.jacrev(func)(pt)
        return jnp.array([(dFx[0,:,0] + dFx[1,:,1]), (dFx[0,:,1] - dFx[1,:,0])])/2.

    delDelBar = localDel(delBarFunc, X)

    return delDelBar

@jax.jit
def manual_det_3x3(M):
    """
    Calculate the determinant of a 3x3 matrix manually.
    Parameters:
    M (numpy.ndarray): A 3x3 matrix represented as a NumPy array.
    Returns:
    float: The determinant of the matrix M.
    Example:
    >>> M = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> manual_det_3x3(M)
    0.0
    """

    det = M[0, 2]*(-M[1, 1]*M[2, 0] + M[1, 0]*M[2, 1]) +\
          M[0, 1]*(M[1, 2]*M[2, 0] - M[1, 0]*M[2, 2]) +\
          M[0, 0]*(-M[1, 2]*M[2, 1] + M[1, 1]*M[2, 2])
    return det