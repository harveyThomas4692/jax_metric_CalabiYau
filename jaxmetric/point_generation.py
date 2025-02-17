import jax.numpy as jnp
import jax
from jax import jit, vmap
from scipy.optimize import root
import numpy as np

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

generate_points_sphere = jit(generate_points_sphere,static_argnums=(1,2))

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

def scale_coordinates_product(pt, projective_factors):
    """
    Scales the coordinates of a point based on given projective factors.
    This function takes a point and a list of projective factors, applies a scaling
    transformation to the point using these factors, and returns the scaled point.
    Args:
        pt (jnp.ndarray): The input point to be scaled. It is expected to be a 1D array.
        projective_factors (tuple): A list of projective factors used for scaling.
    Returns:
        jnp.ndarray: The scaled point as a 1D array.
    """
    # Calculate cumulative sum of splits for splitting the point array
    splits2 = np.cumsum(np.array(projective_factors)+1)[:-1]
    #prods = jnp.array(projective_factors)+1
    splits_pt = jnp.split(pt, splits2)
    scaled = []
    for i in range(len(splits_pt)):
        scaled.append(scale_coordinates(splits_pt[i]))
    return jnp.concatenate(scaled)

scale_coordinates_product = jit(scale_coordinates_product,static_argnums=(1,))

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
    complex_points = points[..., ::2] + 1j * points[..., 1::2]
    return vmap(scale_coordinates)(complex_points)

generate_points_projective = jit(generate_points_projective,static_argnums=(1,2))

def generate_points_projective_product(key, projective_factors, m):
    """
    Generate points in a product of projective spaces, with fixed projective factors for jit compatibility.
    Args:
        key (jax.random.PRNGKey): A random key used for generating random numbers.
        projective_factors (tuple): A tuple of integers where each integer represents 
            the dimension of a projective space.
        m (int): The number of points to generate in each projective space.
    Returns:
        jnp.ndarray: An array of shape (m, sum(projective_factors)) containing the generated points.
    Note:
        This version is jittable since projective_factors is a static argument.
    """
    keys = jax.random.split(key, len(projective_factors))
    points = []
    for i in range(len(projective_factors)):
        points_i = generate_points_projective(keys[i], projective_factors[i], m)
        points.append(points_i)
    points = jnp.concatenate(points, axis=1)
    return points

generate_points_projective_product = jit(generate_points_projective_product, static_argnums=(1,2))

@jit
def compute_line(points, t):
    """
    Computes a point on a line given two points and a parameter t.
    Args:
        points (jnp.ndarray): A 2D array containing two points in projective space.
        t (float): A scalar parameter between 0 and 1.
    Returns:
        jnp.ndarray: A point on the line defined by the two input points.
    """

    return (1-t)*points[0] + t*points[1]

def __toSolve(x,linePts,pol):
    p = pol(compute_line(linePts,x[0] + 1j*x[1]))
    return p.real, p.imag

__toSolve = jit(__toSolve,static_argnums=(2))

def generate_points_calabi_yau(key, projective_factors, pol, m,safe_fac = 1.2):
    """
    Generates points on a Calabi-Yau manifold using projective factors and polynomial equations.
    Args:
        key (jax.random.PRNGKey): Random key for generating points.
        projective_factors (tuple): List of projective factors for the manifold.
        pol (function): Polynomial function defining the Calabi-Yau manifold.
        m (int): Number of points to generate.
        safe_fac (int): Safety factor for mulitple of number of points to generate.
    Returns:
        jax.numpy.ndarray: Array of generated points on the Calabi-Yau manifold.
    Note:
        This function currently doesn't work with JIT compilation due to the loop.
        It is currently very very slow. We should consider optimising it.
        This is currently setup in a way that it may not generate all the points.
        Should consider using using jnp.roots - but this will require some reworking
    """
    
    pair_points = generate_points_projective_product(key, projective_factors, 2*int(m*safe_fac))
    pair_points = pair_points.reshape(int(m*safe_fac),2,sum(projective_factors)+len(projective_factors))
    points = []
    i = 0
    m_orig = m
    while i < m:
        if(i % jnp.round(m/10) == 0 or i == m-1):
            print(f"{i+1}/{m}| Extra needed so far: {m-m_orig}")
        sols = root((lambda x: __toSolve(x,pair_points[i],pol)), [0., 0.],tol=1e-5)
        pt = compute_line(pair_points[i],sols.x[0] + 1.j*sols.x[1])
        pt = scale_coordinates_product(pt, projective_factors)
        if jnp.abs(pol(pt)) < 1e-5:
            points.append(pt)
        else:
            m+=1
        i += 1
    if len(points) < m_orig:
        print("Warning: Not all points generated")
    return jnp.array(points)

#generate_points_calabi_yau = jit(generate_points_calabi_yau, static_argnums=(0,3))