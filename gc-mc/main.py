from data_loader import *
from model import *
from layers import *
from metrics import *
from keras import backend as K
from keras.layers import Concatenate, Input, Dense
from keras.models import Model

import matplotlib.pyplot as plt
import tensorflow as tf

FEATHIDDEN = 64
HIDDEN = [41, 41, 10]
BASES = 2
DROPOUT = 0.7

def softmax_cross_entropy(y_true, y_pred):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
    return K.mean(loss)

# Load the datas
num_users, num_items, num_classes, num_side_features, num_features, u_features,\
v_features, u_features_nonzero, v_features_nonzero, u_features_side, v_features_side,\
normalized_train, normalized_t_train, train_labels, train_u_indices, train_v_indices, adj_train,\
normalized_val, normalized_t_val, val_labels, val_u_indices, val_v_indices, adj_val,\
normalized_test, normalized_t_test, test_labels, test_u_indices, test_v_indices, adj_test = get_loader()

u_features_side = K.variable(u_features_side)
v_features_side = K.variable(v_features_side)

u_inputs = Input(shape=(2,), sparse=True, name="u_inputs")
v_inputs = Input(shape=(2,), sparse=True, name="v_inputs")
u_dense_inputs = Input(shape=(273,), sparse=True, name="u_dense_inputs")
v_dense_inputs = Input(shape=(273,), sparse=True, name="v_dense_inputs")

inputs = [u_inputs, v_inputs, u_dense_inputs, v_dense_inputs]

u_sgc = SGConv(input_dim=num_users,
                output_dim=HIDDEN[0],
                features_nonzero=v_features_nonzero,
                normalized=normalized_train,
                num_classes=num_classes,
                sparse_inputs=True,
                dropout=DROPOUT,
                activation='relu',
                kernel_initializer='he_normal')(inputs[0])
v_sgc = SGConv(input_dim=num_items,
                output_dim=HIDDEN[0],
                features_nonzero=u_features_nonzero,
                normalized=normalized_t_train,
                num_classes=num_classes,
                sparse_inputs=True,
                dropout=DROPOUT,
                activation='relu',
                kernel_initializer='he_normal')(inputs[1])

u_dense1 = Dense(HIDDEN[1], activation='relu')(inputs[2])
v_dense1 = Dense(HIDDEN[1], activation='relu')(inputs[3])

u_m = Concatenate(axis=1)([u_sgc, u_dense1])
v_m = Concatenate(axis=1)([v_sgc, v_dense1])

u_dense2 = Dense(HIDDEN[2], activation='relu')(u_m)
v_dense2 = Dense(HIDDEN[2], activation='relu')(v_m)

bilin_dec = BilinearMixture(num_classes=num_classes,
                            u_indices=train_u_indices,
                            v_indices=train_v_indices,
                            input_dim=HIDDEN[2],
                            num_users=num_users,
                            num_items=num_items,
                            user_item_bias=False,
                            activation='softmax',
                            kernel_initializer='he_normal',
                            bias_initializer='zeros',
                            num_weights=2,
                            diagonal=False)([u_dense2, v_dense2])

model = Model(inputs=inputs,
            outputs=bilin_dec)

model.compile(loss=softmax_cross_entropy,
            optimizer='adam',
            metrics=['accuracy'])
model.summary()

hist = model.fit({'u_inputs': u_features, 'v_inputs': v_features,
                'u_dense_inputs': u_features_side, 'v_dense_inputs': v_features_side},
                train_labels,
                epochs=200)
