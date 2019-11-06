import tensorflow as tf
import numpy as np

def weight_variable_random_uniform(input_dim, output_dim=None, name=""):
    if output_dim is not None:
        init_range = np.sqrt(6.0 / (input_dim + output_dim))
        initial = tf.random_uniform([input_dim, output_dim], minval=-init_range, maxval=init_range, dtype=tf.float32)
    else:
        init_range = np.sqrt(6.0 / input_dim)
        initial = tf.random_uniform([input_dim], minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def bias_variable_truncated_normal(shape, name=""):
    initial = tf.truncated_normal(shape, stddev=0.5)
    return tf.Variable(initial, name=name)

def bias_variable_zero(shape, name=""):
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def orthogonal(shape, scale=1.1, name=None):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)

    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return tf.Variable(scale * q[:shape[0], :shape[1]], name=name, dtype=tf.float32)
