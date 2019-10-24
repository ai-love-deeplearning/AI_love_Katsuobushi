
import numpy as np

def orthogonal(shape, scale=1.1, name=None):

    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)

    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)

    return scale * q[:shape[0], :shape[1]], name
