import time

from data_loader import *
from model_tf import *
from preprocessing import *

import matplotlib.pyplot as plt
import tensorflow as tf

HIDDEN = [410, 82]
FEATHIDDEN = 64
BASES = 2
LR = 0.01
DROPOUT = 0.7
NB_EPOCH = 5
VERBOSE = True
TESTING = True

def construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero, v_features_nonzero,
                        normalized, normalized_t, labels, u_indices, v_indices, class_values,
                        dropout, u_features_side=None, v_features_side=None):
    feed_dict = dict()
    feed_dict.update({placeholders['u_features']: u_features})
    feed_dict.update({placeholders['v_features']: v_features})
    feed_dict.update({placeholders['u_features_nonzero']: u_features_nonzero})
    feed_dict.update({placeholders['v_features_nonzero']: v_features_nonzero})
    feed_dict.update({placeholders['normalized']: normalized})
    feed_dict.update({placeholders['normalized_t']: normalized_t})

    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['user_indices']: u_indices})
    feed_dict.update({placeholders['item_indices']: v_indices})

    feed_dict.update({placeholders['dropout']: dropout})
    feed_dict.update({placeholders['class_values']: class_values})

    if (u_features_side is not None) and (v_features_side is not None):
        feed_dict.update({placeholders['u_features_side']: u_features_side})
        feed_dict.update({placeholders['v_features_side']: v_features_side})

    return feed_dict

seed = int(time.time())
np.random.seed(seed)
tf.set_random_seed(seed)

# Load the datas
num_users, num_items, class_values, num_side_features, num_normalized, u_features, v_features,\
train_u_features_side, train_v_features_side, train_normalized, train_normalized_t,\
train_u_indices, train_v_indices, train_labels,\
val_u_features_side, val_v_features_side, val_normalized, val_normalized_t,\
val_u_indices, val_v_indices, val_labels,\
test_u_features_side, test_v_features_side, test_normalized, test_normalized_t, \
test_u_indices, test_v_indices, test_labels = get_loader()

placeholders = {
    'u_features': tf.sparse_placeholder(tf.float32, shape=np.array(u_features.shape, dtype=np.int64)),
    'v_features': tf.sparse_placeholder(tf.float32, shape=np.array(v_features.shape, dtype=np.int64)),
    'u_features_nonzero': tf.placeholder(tf.int32, shape=()),
    'v_features_nonzero': tf.placeholder(tf.int32, shape=()),
    'labels': tf.placeholder(tf.int32, shape=(None,)),

    'u_features_side': tf.placeholder(tf.float32, shape=(None, num_side_features)),
    'v_features_side': tf.placeholder(tf.float32, shape=(None, num_side_features)),

    'user_indices': tf.placeholder(tf.int32, shape=(None,)),
    'item_indices': tf.placeholder(tf.int32, shape=(None,)),

    'class_values': tf.placeholder(tf.float32, shape=class_values.shape),

    'dropout': tf.placeholder_with_default(0., shape=()),
    'weight_decay': tf.placeholder_with_default(0., shape=()),

    'normalized': tf.sparse_placeholder(tf.float32, shape=(None, None)),
    'normalized_t': tf.sparse_placeholder(tf.float32, shape=(None, None)),
}

model = GAE(placeholders,
            input_dim=u_features.shape[1],
            feat_hidden_dim=FEATHIDDEN,
            num_classes=len(class_values),
            num_normalized=num_normalized,
            num_basis_functions=BASES,
            hidden=HIDDEN,
            num_users=num_users,
            num_items=num_items,
            learning_rate=LR,
            num_side_features=num_side_features,
            logging=True)

test_normalized = sparse_to_tuple(test_normalized)
test_normalized_t = sparse_to_tuple(test_normalized_t)

val_normalized = sparse_to_tuple(val_normalized)
val_normalized_t = sparse_to_tuple(val_normalized_t)

train_normalized = sparse_to_tuple(train_normalized)
train_normalized_t = sparse_to_tuple(train_normalized_t)

u_features = sparse_to_tuple(u_features)
v_features = sparse_to_tuple(v_features)

assert u_features[2][1] == v_features[2][1], 'Number of features of users and items must be the same!'

num_features = u_features[2][1]
u_features_nonzero = u_features[1].shape[0]
v_features_nonzero = v_features[1].shape[0]

train_feed_dict = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                      v_features_nonzero, train_normalized, train_normalized_t,
                                      train_labels, train_u_indices, train_v_indices, class_values, DROPOUT,
                                      train_u_features_side, train_v_features_side)

val_feed_dict = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                    v_features_nonzero, val_normalized, val_normalized_t,
                                    val_labels, val_u_indices, val_v_indices, class_values, 0.,
                                    val_u_features_side, val_v_features_side)

test_feed_dict = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                     v_features_nonzero, test_normalized, test_normalized_t,
                                     test_labels, test_u_indices, test_v_indices, class_values, 0.,
                                     test_u_features_side, test_v_features_side)

merged_summary = tf.summary.merge_all()

sess = tf.Session()
sess.run(tf.global_variables_initializer()) # tf.Variable()を使う際の初期化

log_dir = './logs'
if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)
tf.gfile.MakeDirs(log_dir)

train_summary_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
val_summary_writer = tf.summary.FileWriter(log_dir + '/val')

# float型の無限大
best_val_score = np.inf
best_val_loss = np.inf
best_epoch = 0
wait = 0

print('Training...')

for epoch in range(NB_EPOCH):
    t = time.time()

    # 学習
    outs = sess.run([model.training_op, model.loss, model.rmse, model.outputs], feed_dict=train_feed_dict)

    train_avg_loss = outs[1]
    train_rmse = outs[2]

    val_avg_loss, val_rmse, val_outputs = sess.run([model.loss, model.rmse, model.outputs], feed_dict=val_feed_dict)

    if VERBOSE:
        print("[*] Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_avg_loss),
              "train_rmse=", "{:.5f}".format(train_rmse),
              "val_loss=", "{:.5f}".format(val_avg_loss),
              "val_rmse=", "{:.5f}".format(val_rmse),
              "\t\ttime=", "{:.5f}".format(time.time() - t))

    if val_rmse < best_val_score:
        best_val_score = val_rmse
        best_epoch = epoch

    # 学習結果を上書き
    if epoch % 200 == 0:
        summary = sess.run(merged_summary, feed_dict=train_feed_dict)
        train_summary_writer.add_summary(summary, epoch)
        train_summary_writer.flush()

        summary = sess.run(merged_summary, feed_dict=val_feed_dict)
        val_summary_writer.add_summary(summary, epoch)
        val_summary_writer.flush()

    # 学習パラメータの保存
    if epoch % 500 == 0 and epoch > 1000 and not TESTING and False:
        saver = tf.train.Saver()
        save_path = saver.save(sess, "./tmp/%s_seed%d.ckpt" % (model.name, DATASEED), global_step=model.global_step)

        variables_to_restore = model.variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, save_path)

        val_avg_loss, val_rmse = sess.run([model.loss, model.rmse], feed_dict=val_feed_dict)

        print('polyak val loss = ', val_avg_loss)
        print('polyak val rmse = ', val_rmse)

        # パラメータ読み込み用処理
        saver = tf.train.Saver()
        saver.restore(sess, save_path)

saver = tf.train.Saver()
save_path = saver.save(sess, "./tmp/%s.ckpt" % model.name, global_step=model.global_step)

if VERBOSE:
    print("\nOptimization Finished!")
    print('best validation score =', best_val_score, 'at iteration', best_epoch)

if TESTING:
    test_avg_loss, test_rmse, test_outputs = sess.run([model.loss, model.rmse, model.outputs], feed_dict=test_feed_dict)
    print('test loss = ', test_avg_loss)
    print('test rmse = ', test_rmse)
    print('@@@@@@@@@@')
    print('test_outputs')
    print(test_outputs.shape)

    variables_to_restore = model.variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, save_path)

print('global seed = ', seed)

sess.close()

# results = []
# row_dict = {i: r for i, r in enumerate(outs[3][0].tolist())}
# row_dict_max = max(row_dict, key=row_dict.get)
# print('@@@@@@@@@@')
# print('outs[3][0]')
# print(outs[3][0])
# print('@@@@@@@@@@')
# print('row_dict_max')
# print(row_dict_max)
#
# row_dict = {i: r for i, r in enumerate(outs[3][150].tolist())}
# row_dict_max = max(row_dict, key=row_dict.get)
# print('@@@@@@@@@@')
# print('outs[3][150]')
# print(outs[3][150])
# print('@@@@@@@@@@')
# print('row_dict_max')
# print(row_dict_max)
