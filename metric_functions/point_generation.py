import jax.numpy as jnp
import jax
from jax import jit, vmap
import multiprocessing as mp

def generate_points_sphere(key,n,m):
    """
    Generates a random point on the surface of an n-dimensional sphere.
    This function uses a normal distribution to generate a point in (n+1)-dimensional space,
    then normalizes the point to lie on the surface of an n-dimensional sphere.
    Parameters:
    key (jax.random.PRNGKey): The random key for generating random numbers.
    n (int): The dimension of the sphere.
    m (int): The number of points to generate.
    Returns:
    jax.numpy.ndarray: m points on the surface of the n-dimensional sphere.
    Example:
    >>> import jax
    >>> key = jax.random.PRNGKey(0)
    >>> generate_point_sphere(key, 2, 1)
    DeviceArray([[ 0.12950151 -0.88474877  0.44771528]], dtype=float32)
    """

    points = jax.random.normal(key,shape=(m,n+1,))

    return vmap(lambda x: x/(jnp.linalg.norm(x)))(points)

generate_point_sphere = jit(generate_points_sphere, static_argnums=(0,))

def generate_points_projective(key,n,m):
    """
    Generates points uniformly on the complex projective space P^n.

    Parameters:
    key (any): A random key or seed used for generating random points.
    n (int): The dimension of the projective space.
    m (int): The number of points to generate.
    Returns:
    jnp.ndarray: An array of complex numbers representing m points on P^n.

    Example:
    >>> key = some_random_key_function()
    >>> point = generate_point_projective(key, 1, 1)
    >>> print(point)
    DeviceArray([[0.04606142-0.22000188j -0.21396911+0.9506286j]], dtype=float32)

    Note:
    This function relies on `generate_point_sphere` to generate points on a sphere.
    """

    points = generate_points_sphere(key,2*(n-1)+3,m)
    return vmap(lambda point: jnp.array([point[i] + 1j*point[i+1] for i in range(0, len(point), 2)]))(points)

generate_point_projective = jit(generate_points_projective, static_argnums=(0,))

def generate_random_lines_projective(key, n, m):
    """
    Generates a random line in projective space.
    This function generates two random points in projective space and returns a 
    lambda function representing a line parameterized by t, where t is a scalar 
    between 0 and 1. The line is defined as a linear interpolation between the 
    two points.
    Args:
        key: A key used for random number generation.
        n: The dimension of the projective space.
        m: The number of lines to generate.
    Returns:
        A list of lambda functions that takes a scalar t and returns a point on the line 
        in projective space.
    """

    points = generate_points_projective(key,n,2*m)
    points = points.reshape(m,2,n+1)
    return [lambda t: (1-t)*points[i,0] + t*points[i,1] for i in range(m)]

@jit
def scale_coordinates(pt):
    """
    Scales the given point such that the coordinate with the largest absolute value becomes 1.
    Parameters:
    pt (array-like): A point represented as an array or list of numerical values.
    Returns:
    array-like: The scaled point where the coordinate with the largest absolute value is scaled to 1.
    Example:
    >>> pt = jnp.array([3, 4, -5])
    >>> scale_coordinates(pt,1)
    DeviceArray([[-0.6, -0.8, 1. ]], dtype=float32)
    """

    arg = jnp.argmax(jnp.abs(pt))
    return pt/pt[arg]

def generate_points_calabi_yau(key, projective_factors, k_moduli, pol, m):
    '''
    Geneates a point a Calabi-Yau manifold, given a set of projective factors and defining polynomial.
    Parameters:
    key (jax.random.PRNGKey): The random key for generating random numbers.
    projective_factors (list): A list of integers representing the projective factors of the Calabi-Yau manifold.
    k_moduli (list): A list of integers representing the k-moduli of the Calabi-Yau manifold.
    pol (function): A function representing the defining polynomial of the Calabi-Yau manifold.
    m (int): The number of points to generate.
    Returns:
    jax.numpy.ndarray: m points on the Calabi-Yau manifold.
    Example:
    >>> key = jax.random.PRNGKey(0)
    >>> projective_factors = [4]
    >>> pol = lambda x : x[0]**5 + x[1]**5 + x[2]**5 + x[3]**5 + x[4]**5
    >>> generate_point_calabi_yau(key, projective_factors, [1], pol, 1)
    DeviceArray([[[0.12950151+0.j -0.88474877+0.j  0.44771528+0.j, 1.+0.j]]], dtype=float32)
    '''

    keys = jax.random.split(key,len(projective_factors))
    points = [jnp.reshape(generate_points_projective(keys[i],projective_factors[i],m*2),(m,2,projective_factors[i]+1)) for i in range(len(projective_factors))]
    
    return points

#generate_points_calabi_yau = jit(generate_points_calabi_yau, static_argnums=(0,3))