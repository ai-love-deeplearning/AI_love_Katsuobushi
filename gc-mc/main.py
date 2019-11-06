from data_loader import *
from model import *
from layers import *
from metrics import *
from preprocessing import *
from keras import backend as K
from keras.layers import Concatenate, Input, Dense
from keras.models import Model

import matplotlib.pyplot as plt
import tensorflow as tf

FEATHIDDEN = 64
HIDDEN = [41, 41, 41]
BASES = 2
DROPOUT = 0.7

def softmax_cross_entropy(y_true, y_pred):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
    return K.mean(loss)

# Load the datas
num_users, num_items, num_classes, num_side_features, u_features, v_features,\
train_u_features_side, train_v_features_side, train_normalized, train_normalized_t,\
train_u_indices, train_v_indices, train_labels,\
val_u_features_side, val_v_features_side, val_normalized, val_normalized_t,\
val_u_indices, val_v_indices, val_labels,\
test_u_features_side, test_v_features_side, test_normalized, test_normalized_t, \
test_u_indices, test_v_indices, test_labels = get_loader()

# データ整形
num_features = u_features.shape[1]
u_features_nonzero = u_features.shape[0]
v_features_nonzero = v_features.shape[0]

u_features = convert_sparse_matrix_to_sparse_tensor(u_features)
v_features = convert_sparse_matrix_to_sparse_tensor(v_features)

train_normalized = convert_sparse_matrix_to_sparse_tensor(train_normalized)
train_normalized_t = convert_sparse_matrix_to_sparse_tensor(train_normalized_t)
train_u_features_side = K.variable(train_u_features_side)
train_v_features_side = K.variable(train_v_features_side)
train_num_labels = len(train_labels)

val_normalized = convert_sparse_matrix_to_sparse_tensor(val_normalized)
val_normalized_t = convert_sparse_matrix_to_sparse_tensor(val_normalized_t)
val_u_features_side = K.variable(val_u_features_side)
val_v_features_side = K.variable(val_v_features_side)
val_num_labels = len(val_labels)

test_normalized = convert_sparse_matrix_to_sparse_tensor(test_normalized)
test_normalized_t = convert_sparse_matrix_to_sparse_tensor(test_normalized_t)
test_u_features_side = K.variable(test_u_features_side)
test_v_features_side = K.variable(test_v_features_side)
test_num_labels = len(test_labels)

u_dense_inputs = Input(shape=(num_side_features,), name="u_dense_inputs")
v_dense_inputs = Input(shape=(num_side_features,), name="v_dense_inputs")

u_sgc = SGConv(input_dim=num_features,
                output_dim=HIDDEN[0],
                features_nonzero=v_features_nonzero,
                normalized=train_normalized,
                num_classes=num_classes,
                sparse_inputs=True,
                dropout=DROPOUT,
                activation='relu',
                kernel_initializer='he_normal')(v_features)
v_sgc = SGConv(input_dim=num_features,
                output_dim=HIDDEN[0],
                features_nonzero=u_features_nonzero,
                normalized=train_normalized_t,
                num_classes=num_classes,
                sparse_inputs=True,
                dropout=DROPOUT,
                activation='relu',
                kernel_initializer='he_normal')(u_features)

u_dense1 = Dense(HIDDEN[1], activation='relu')(u_dense_inputs)
v_dense1 = Dense(HIDDEN[1], activation='relu')(v_dense_inputs)

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
                            num_labels=train_num_labels,
                            user_item_bias=False,
                            kernel_initializer='he_normal',
                            bias_initializer='zeros',
                            num_weights=2,
                            diagonal=False)([u_dense2, v_dense2])

predictions = Dense(HIDDEN[2], activation='softmax')(bilin_dec)

model = Model(inputs=[u_dense_inputs, v_dense_inputs],
            outputs=predictions)

model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
model.summary()

# hist = model.fit(train_u_features_side,
#                 train_labels,
#                 epochs=200)
