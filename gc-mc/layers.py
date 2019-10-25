
from keras import backend as K
from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine.topology import Layer
from initializations import *
import tensorflow as tf
import numpy as np
import scipy.sparse as sp

#kernel_initializer='glorot_uniform'は、活性化関数が原点対称のとき
#reluで活性化する時には kernel_initializer='he_normal' を使うらしい

def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

# keep_prob: (1 - dropout_rate)
def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """
    Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    スパーステンソルのドロップアウト。 現在、非常に大きなスパーステンソル（> 1M要素）で失敗します
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    pre_out = tf.cast(pre_out, tf.float32)

    return pre_out * tf.div(1., keep_prob)

class SGConv(Layer):

    def __init__(self, input_dim, output_dim, normalized, num_classes,
                features_nonzero=None, sparse_inputs=False, dropout=0.,
                activation=None, kernel_initializer='he_normal', **kwargs):

        super(SGConv, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.sparse_inputs = sparse_inputs
        self.features_nonzero = features_nonzero
        # self.u_features_nonzero = u_features_nonzero
        # self.v_features_nonzero = v_features_nonzero
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self._weights = None

        self.normalized = tf.sparse_split(axis=1, num_split=num_classes, sp_input=normalized)
        # self.normalized_t = tf.sparse_split(axis=1, num_split=num_classes, sp_input=normalized_t)

        self.num_classes = num_classes

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        self._weights = tf.split(value=self.kernel, axis=1, num_or_size_splits=self.num_classes)

        super(SGConv, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, input):
        x = input
        # x_u = inputs[0]
        # x_v = inputs[1]

        if self.sparse_inputs:
            x = dropout_sparse(x, 1 - self.dropout, self.features_nonzero.value)
            # x_u = dropout_sparse(x_u, 1 - self.dropout, self.u_features_nonzero.value)
            # x_v = dropout_sparse(x_v, 1 - self.dropout, self.v_features_nonzero.value)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)
            # x_u = tf.nn.dropout(x_u, 1 - self.dropout)
            # x_v = tf.nn.dropout(x_v, 1 - self.dropout)

        normalizeds = []
        # normalized_u = []
        # normalized_v = []

        for i in range(len(self.normalized)):
            tmp = dot(x, self._weights[i], sparse=self.sparse_inputs)
            # tmp_u = dot(x_u, self.weights_u[i], sparse=self.sparse_inputs)
            # tmp_v = dot(x_v, self.weights_v[i], sparse=self.sparse_inputs)

            normalized = self.normalized[i]
            # normalized_t = self.normalized_t[i]

            normalizeds.append(tf.sparse_tensor_dense_matmul(normalized, tmp))
            # normalized_u.append(tf.sparse_tensor_dense_matmul(normalized, tmp_v))
            # normalized_v.append(tf.sparse_tensor_dense_matmul(normalized_t, tmp_u))

        # 分割と結合するのにh方向かd方向か要検討
        z = tf.concat(axis=1, values=normalizeds)
        # z_u = tf.concat(axis=1, values=normalized_u)
        # z_v = tf.concat(axis=1, values=normalized_v)

        outputs = self.activation(z)
        # u_outputs = self.activation(z_u)
        # v_outputs = self.activation(z_v)

        return outputs
        # return u_outputs, v_outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

class BilinearMixture(Layer):

    def __init__(self, num_classes, u_indices, v_indices, input_dim, num_users, num_items, user_item_bias=False,
                activation=None, kernel_initializer='he_normal', bias_initializer='zeros', num_weights=3, diagonal=True,
                **kwargs):

        super(BilinearMixture, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_users = num_users
        self.num_items = num_items
        self.num_weights = num_weights
        self.user_item_bias = user_item_bias
        self.diagonal = diagonal

        if diagonal:
            self._multiply_inputs_weights = np.multiply
        else:
            self._multiply_inputs_weights = np.dot

        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.u_indices = u_indices
        self.v_indices = v_indices

    def build(self, input_shape):
        for i in range(self.num_weights):
            if self.diagonal:
                self.kernel = self.add_weight(name='weights_%d' % i,
                                            shape=(1, self.input_dim),
                                            initializer=self.kernel_initializer,
                                            trainable=True)
            else:
                # w_shape, w_name = orthogonal([self.input_dim, self.input_dim], name='weights_%d' % i)
                self.kernel = self.add_weight(name='weights_%d' % i,
                                            shape=[self.input_dim, self.input_dim],
                                            initializer=self.kernel_initializer,
                                            trainable=True)
        self.kernel = self.add_weight(name='weights_u_scalars',
                                    shape=(self.num_weights, self.num_classes),
                                    initializer=self.kernel_initializer,
                                    trainable=True)

        self.u_bias = self.add_weight(name='bias',
                                    shape=(self.num_users, self.num_classes),
                                    initializer=self.bias_initializer)
        self.v_bias = self.add_weight(name='bias',
                                    shape=(self.num_items, self.num_classes),
                                    initializer=self.bias_initializer)

        super(BilinearMixture, self).build(input_shape)

    def _call(self, inputs):
        u_inputs = np.take(inputs[0], self.u_indices)
        v_inputs = np.take(v_inputs, self.v_indices)

        if self.user_item_bias:
            u_bias = np.take(self.u_bias, self.u_indices)
            v_bias = np.take(self.v_bias, self.v_indices)
        else:
            u_bias = None
            v_bias = None

        basis_outputs = []
        for weight in self.weights:
            u_w = self._multiply_inputs_weights(u_inputs, weight)
            x = np.sum(np.multiply(u_w, v_inputs), axis=1)

            basis_outputs.append(x)

        basis_outputs = np.stack(basis_outputs, axis=1)

        outputs = np.dot(basis_outputs, self.kernel['weights_scalars'])

        if self.user_item_bias:
            outputs += u_bias
            outputs += v_bias

        outputs = self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
