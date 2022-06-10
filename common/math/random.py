"""Functions for random sampling"""
import numpy as np
from scipy.spatial.transform import Rotation

def generate_rand_rotm(x_lim=5.0, y_lim=5.0, z_lim=180.0):
    '''
    Input:
        x_lim
        y_lim
        z_lim
    return:
        rotm: [3,3]
    '''
    rand_z = np.random.uniform(low=-z_lim, high=z_lim)
    rand_y = np.random.uniform(low=-y_lim, high=y_lim)
    rand_x = np.random.uniform(low=-x_lim, high=x_lim)

    rand_eul = np.array([rand_z, rand_y, rand_x])
    r = Rotation.from_euler('zyx', rand_eul, degrees=True)
    rotm = r.as_matrix()
    return rotm

def generate_rand_trans(x_lim=10.0, y_lim=1.0, z_lim=0.1):
    '''
    Input:
        x_lim
        y_lim
        z_lim
    return:
        trans [3]
    '''
    rand_x = np.random.uniform(low=-x_lim, high=x_lim)
    rand_y = np.random.uniform(low=-y_lim, high=y_lim)
    rand_z = np.random.uniform(low=-z_lim, high=z_lim)

    rand_trans = np.array([rand_x, rand_y, rand_z])

    return rand_trans


def uniform_2_sphere(num: int = None):
    """Uniform sampling on a 2-sphere

    Source: https://gist.github.com/andrewbolster/10274979

    Args:
        num: Number of vectors to sample (or None if single)

    Returns:
        Random Vector (np.ndarray) of size (num, 3) with norm 1.
        If num is None returned value will have size (3,)

    """
    if num is not None:
        phi = np.random.uniform(0.0, 2 * np.pi, num)
        cos_theta = np.random.uniform(-1.0, 1.0, num)
    else:
        phi = np.random.uniform(0.0, 2 * np.pi)
        cos_theta = np.random.uniform(-1.0, 1.0)

    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack((x, y, z), axis=-1)


