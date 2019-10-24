
import keras
import tensorflow as tf
from layers import *
from metrics import *

class GAE(keras.Model):

    def __init__(self, u_features, v_features, u_features_nonzero, v_features_nonzero,
                u_features_side, v_features_side, input_dim, feat_hidden_dim, hidden,
                normalized, normalized_t, num_users, num_items, num_basis_functions,
                num_classes, num_side_features, u_indices, v_indices, dropout):

        super(GAE, self).__init__()

        self.outputs = None
        self.u_features_side = u_features_side
        self.v_features_side = v_features_side
        
        self.activations = []
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

        self.sgc = SGConv(input_dim=input_dim,
                        output_dim=hidden[0],
                        u_features_nonzero=u_features_nonzero,
                        v_features_nonzero=v_features_nonzero,
                        normalized=normalized,
                        normalized_t=normalized_t,
                        num_classes=num_classes,
                        sparse_inputs=True,
                        dropout=dropout,
                        activation='relu',
                        kernel_initializer='he_normal')
        self.dense1 = keras.layers.Dense(input_dim=num_side_features,
                                        units=feat_hidden_dim,
                                        activation='relu',
                                        use_bias=True)
        self.dense2 = keras.layers.Dense(input_dim=feat_hidden_dim+hidden[0],
                                        units=hidden[1],
                                        activation='relu',
                                        use_bias=True)
        self.bilin_dec = BilinearMixture(num_classes=num_classes,
                                        u_indices=u_indices,
                                        v_indices=v_indices,
                                        input_dim=hidden[1],
                                        num_users=num_users,
                                        num_items=num_items,
                                        user_item_bias=False,
                                        activation='softmax',
                                        kernel_initializer='he_normal',
                                        bias_initializer='zeros',
                                        num_weights=num_basis_functions,
                                        diagonal=False)

    def call(self, inputs):

        # gcn layer
        layer = self.sgc
        gcn_hidden = layer(inputs)

        # dense layer for features
        layer = self.dense1
        feat_hidden = layer([self.u_features_side, self.v_features_side])

        # concat dense layer
        layer = self.dense2

        gcn_u = gcn_hidden[0]
        gcn_v = gcn_hidden[1]
        feat_u = feat_hidden[0]
        feat_v = feat_hidden[1]

        input_u = tf.concat(values=[gcn_u, feat_u], axis=1)
        input_v = tf.concat(values=[gcn_v, feat_v], axis=1)

        concat_hidden = layer([input_u, input_v])

        self.activations.append(concat_hidden)

        for layer in self.bilin_dec:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        self._accuracy()

        return self.outputs, self.accuracy
