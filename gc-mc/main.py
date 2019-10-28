from data_loader import *
from model import *
from keras import backend as K

FEATHIDDEN = 64
HIDDEN = [41, 41]
BASES = 2
DROPOUT = 0.7

def softmax_cross_entropy(y_true, y_pred):
    print('@@@@@')
    print(y_true)
    print(y_pred)
    print('@@@@@')
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
    return K.mean(loss)

# Load the datas
num_users, num_items, num_classes, num_side_features, num_features, u_features,\
v_features, u_features_nonzero, v_features_nonzero, u_features_side, v_features_side,\
normalized_train, normalized_t_train, train_labels, train_u_indices, train_v_indices, adj_train,\
normalized_val, normalized_t_val, val_labels, val_u_indices, val_v_indices, adj_val,\
normalized_test, normalized_t_test, test_labels, test_u_indices, test_v_indices, adj_test = get_loader()

u_features_side = K.variable(u_features_side)
v_features_side = K.variable(v_features_side)
adj_train = K.variable(adj_train)
print('@@@@@@@@@@@@@@@@@@')
print([u_features, v_features])
print('@@@@@@@@@@@@@@@@@@')

# Creating the architecture of the Neural Network
model = GAE(u_features=u_features,
            v_features=v_features,
            u_features_nonzero=u_features_nonzero,
            v_features_nonzero=v_features_nonzero,
            u_features_side=u_features_side,
            v_features_side=v_features_side,
            input_dim=num_features,
            feat_hidden_dim=FEATHIDDEN,
            hidden=HIDDEN,
            normalized=normalized_train,
            normalized_t=normalized_t_train,
            num_users=num_users,
            num_items=num_items,
            num_basis_functions=BASES,
            num_classes=num_classes,
            num_side_features=num_side_features,
            u_indices=train_u_indices,
            v_indices=train_v_indices,
            dropout=DROPOUT)

model.compile(loss=softmax_cross_entropy,
            optimizer='adam',
            metrics=['accuracy'])

hist = model.fit([u_features, v_features],
                adj_train)
