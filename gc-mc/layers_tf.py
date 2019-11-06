import tensorflow as tf
from initializations import *

_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def dropout_sparse(x, keep_prob, num_nonzero_elems):
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)

    return pre_out * tf.div(1., keep_prob)

def dot(x, y, sparse=False):
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

class Layer(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

class Dense(Layer):
    def __init__(self, input_dim, output_dim, dropout=0., act=tf.nn.relu,
                share_user_item_weights=False, bias=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        with tf.variable_scope(self.name + '_vars'):
            if not share_user_item_weights:
                self.vars['weights_u'] = weight_variable_random_uniform(input_dim, output_dim, name="weights_u")
                self.vars['weights_v'] = weight_variable_random_uniform(input_dim, output_dim, name="weights_v")

                if bias:
                    self.vars['user_bias'] = bias_variable_truncated_normal([output_dim], name="bias_u")
                    self.vars['item_bias'] = bias_variable_truncated_normal([output_dim], name="bias_v")
            else:
                self.vars['weights_u'] = weight_variable_random_uniform(input_dim, output_dim, name="weights")
                self.vars['weights_v'] = self.vars['weights_u']

                if bias:
                    self.vars['user_bias'] = bias_variable_truncated_normal([output_dim], name="bias_u")
                    self.vars['item_bias'] = self.vars['user_bias']

        self.bias = bias

        self.dropout = dropout
        self.act = act
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x_u = inputs[0]
        x_u = tf.nn.dropout(x_u, 1 - self.dropout)
        x_u = tf.matmul(x_u, self.vars['weights_u'])

        x_v = inputs[1]
        x_v = tf.nn.dropout(x_v, 1 - self.dropout)
        x_v = tf.matmul(x_v, self.vars['weights_v'])

        u_outputs = self.act(x_u)
        v_outputs = self.act(x_v)

        if self.bias:
            u_outputs += self.vars['user_bias']
            v_outputs += self.vars['item_bias']

        return u_outputs, v_outputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging:
                tf.summary.histogram(self.name + '/inputs_u', inputs[0])
                tf.summary.histogram(self.name + '/inputs_v', inputs[1])
            outputs_u, outputs_v = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs_u', outputs_u)
                tf.summary.histogram(self.name + '/outputs_v', outputs_v)
            return outputs_u, outputs_v

class StackGCN(Layer):
    def __init__(self, input_dim, output_dim, normalized, normalized_t, num_normalized,
                u_features_nonzero=None, v_features_nonzero=None, sparse_inputs=False,
                dropout=0., act=tf.nn.relu, share_user_item_weights=True, **kwargs):
        super(StackGCN, self).__init__(**kwargs)

        assert output_dim % num_normalized == 0, 'output_dim must be multiple of num_normalized for stackGC layer'

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_u'] = weight_variable_random_uniform(input_dim, output_dim, name='weights_u')

            if not share_user_item_weights:
                self.vars['weights_v'] = weight_variable_random_uniform(input_dim, output_dim, name='weights_v')
            else:
                self.vars['weights_v'] = self.vars['weights_u']

        self.weights_u = tf.split(value=self.vars['weights_u'], axis=1, num_or_size_splits=num_normalized)
        self.weights_v = tf.split(value=self.vars['weights_v'], axis=1, num_or_size_splits=num_normalized)

        self.dropout = dropout

        self.sparse_inputs = sparse_inputs
        self.u_features_nonzero = u_features_nonzero
        self.v_features_nonzero = v_features_nonzero
        if sparse_inputs:
            assert u_features_nonzero is not None and v_features_nonzero is not None, \
                'u_features_nonzero and v_features_nonzero can not be None when sparse_inputs is True'

        self.normalized = tf.sparse_split(axis=1, num_split=num_normalized, sp_input=normalized)
        self.normalized_transpose = tf.sparse_split(axis=1, num_split=num_normalized, sp_input=normalized_t)

        self.act = act

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x_u = inputs[0]
        x_v = inputs[1]

        if self.sparse_inputs:
            x_u = dropout_sparse(x_u, 1 - self.dropout, self.u_features_nonzero)
            x_v = dropout_sparse(x_v, 1 - self.dropout, self.v_features_nonzero)
        else:
            x_u = tf.nn.dropout(x_u, 1 - self.dropout)
            x_v = tf.nn.dropout(x_v, 1 - self.dropout)

        normalizeds_u = []
        normalizeds_v = []

        for i in range(len(self.normalized)):
            tmp_u = dot(x_u, self.weights_u[i], sparse=self.sparse_inputs)
            tmp_v = dot(x_v, self.weights_v[i], sparse=self.sparse_inputs)

            normalized = self.normalized[i]
            normalized_transpose = self.normalized_transpose[i]

            normalizeds_u.append(tf.sparse_tensor_dense_matmul(normalized, tmp_v))
            normalizeds_v.append(tf.sparse_tensor_dense_matmul(normalized_transpose, tmp_u))

        z_u = tf.concat(axis=1, values=normalizeds_u)
        z_v = tf.concat(axis=1, values=normalizeds_v)

        u_outputs = self.act(z_u)
        v_outputs = self.act(z_v)

        return u_outputs, v_outputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs_u', inputs[0])
                tf.summary.histogram(self.name + '/inputs_v', inputs[1])
            outputs_u, outputs_v = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs_u', outputs_u)
                tf.summary.histogram(self.name + '/outputs_v', outputs_v)
            return outputs_u, outputs_v

class BilinearMixture(Layer):
    def __init__(self, num_classes, u_indices, v_indices, input_dim, num_users,
                num_items, user_item_bias=False, dropout=0., act=tf.nn.softmax,
                num_weights=3, diagonal=True, **kwargs):
        super(BilinearMixture, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            for i in range(num_weights):
                if diagonal:
                    self.vars['weights_%d' % i] = weight_variable_random_uniform(1, input_dim, name='weights_%d' % i)
                else:
                    self.vars['weights_%d' % i] = orthogonal([input_dim, input_dim], name='weights_%d' % i)

            self.vars['weights_scalars'] = weight_variable_random_uniform(num_weights, num_classes,
                                                                          name='weights_u_scalars')
            if user_item_bias:
                self.vars['user_bias'] = bias_variable_zero([num_users, num_classes], name='user_bias')
                self.vars['item_bias'] = bias_variable_zero([num_items, num_classes], name='item_bias')

        self.user_item_bias = user_item_bias

        if diagonal:
            self._multiply_inputs_weights = tf.multiply
        else:
            self._multiply_inputs_weights = tf.matmul

        self.num_classes = num_classes
        self.num_weights = num_weights
        self.u_indices = u_indices
        self.v_indices = v_indices

        self.dropout = dropout
        self.act = act
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        u_inputs = tf.nn.dropout(inputs[0], 1 - self.dropout)
        v_inputs = tf.nn.dropout(inputs[1], 1 - self.dropout)

        u_inputs = tf.gather(u_inputs, self.u_indices)
        v_inputs = tf.gather(v_inputs, self.v_indices)

        if self.user_item_bias:
            u_bias = tf.gather(self.vars['user_bias'], self.u_indices)
            v_bias = tf.gather(self.vars['item_bias'], self.v_indices)
        else:
            u_bias = None
            v_bias = None

        basis_outputs = []
        for i in range(self.num_weights):
            u_w = self._multiply_inputs_weights(u_inputs, self.vars['weights_%d' % i])
            x = tf.reduce_sum(tf.multiply(u_w, v_inputs), axis=1)

            basis_outputs.append(x)

        basis_outputs = tf.stack(basis_outputs, axis=1)

        outputs = tf.matmul(basis_outputs,  self.vars['weights_scalars'], transpose_b=False)

        if self.user_item_bias:
            outputs += u_bias
            outputs += v_bias

        outputs = self.act(outputs)

        return outputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs_u', inputs[0])
                tf.summary.histogram(self.name + '/inputs_v', inputs[1])
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs
