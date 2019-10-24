
from keras import backend as K
from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine.topology import Layer
import numpy as np
import scipy.sparse as sp
​
#kernel_initializer='glorot_uniform'は、活性化関数が原点対称のとき
#reluで活性化する時には kernel_initializer='he_normal' を使うらしい
​
class SGConv(Layer):

    def __init__(self, output_dim, normalized, normalized_t, num_classes, sparse_inputs=False, activation=None,
                kernel_initializer='he_normal', **kwargs):

        super(SGConv, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.sparse_inputs = sparse_inputs
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)

        self.normalized = np.dsplit(normalized, num_classes)
        self.normalized_t = np.dsplit(normalized_t, num_classes)
​
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel_u = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=self.kernel_initializer,
                                      trainable=True)

        self.kernel_v = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=self.kernel_initializer,
                                      trainable=True)

        self.weights_u = np.dsplit(self.kernel_u, self.num_classes)
        self.weights_v = np.dsplit(self.kernel_v, self.num_classes)

        super(SGConv, self).build(input_shape)  # Be sure to call this somewhere!
​
    def call(self, inputs):
        x_u = inputs[0]
        x_v = inputs[1]

        normalized_u = []
        normalized_v = []

        for i in range(len(self.normalized)):
            tmp_u = dot(x_u, self.weights_u[i], sparse=self.sparse_inputs)
            tmp_v = dot(x_v, self.weights_v[i], sparse=self.sparse_inputs)

            normalized = self.normalized[i]
            normalized_t = self.normalized_t[i]

            normalized_u.append(normalized.dot(tmp_v).toscr())
            normalized_v.append(normalized_t.dot(tmp_u).toscr())

        # 分割と結合するのにh方向かd方向か要検討
        z_u = np.dstack(normalized_u)
        z_v = np.dstack(normalized_v)

        u_outputs = self.activation(z_u)
        v_outputs = self.activation(z_v)

        return u_outputs, v_outputs
​
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

class BilinearMixture(Layer):

    def __init__(self, num_classes, u_indices, v_indices, input_dim, num_users, num_items, user_item_bias=False,
                , activation=None, kernel_initializer='he_normal', bias_initializer='zeros', num_weights=3, diagonal=True
                , **kwargs):

        super(BilinearMixture, self).__init__(**kwargs)

        self.num_classes = num_classes
        self.num_users = num_users
        self.num_items = num_items
        self.num_weights = num_weights
        self.user_item_bias = user_item_bias

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
            if diagonal:
                self.kernel['weights_%d' % i] = self.add_weight(name='weights_%d' % i,
                                                                shape=(1, self.input_dim),
                                                                initializer=self.kernel_initializer,
                                                                trainable=True)
            else:
                shape, name = orthogonal([input_dim, input_dim], name='weights_%d' % i)
                self.kernel['weights_%d' % i] = self.add_weight(name=name,
                                                                shape=shape,
                                                                initializer=self.kernel_initializer,
                                                                trainable=True)
        self.kernel['weights_scalars'] = self.add_weight(name='weights_u_scalars',
                                                        shape=(num_weights, num_classes),
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
