
import tensorflow as tf

def softmax_accuracy(preds, labels):
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.to_int64(labels))
    accuracy_all = tf.cast(correct_prediction, tf.float32)

    return tf.reduce_mean(accuracy_all)

def softmax_cross_entropy(outputs, labels):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=labels)
    return tf.reduce_mean(loss)

def expected_rmse(logits, labels, class_values=None):
    probs = tf.nn.softmax(logits)
    if class_values is None:
        scores = tf.to_float(tf.range(start=0, limit=logits.get_shape()[1]) + 1)
        y = tf.to_float(labels) + 1.  # assumes class values are 1, ..., num_classes
    else:
        scores = class_values
        y = tf.gather(class_values, labels)

    pred_y = tf.reduce_sum(probs * scores, 1)

    diff = tf.subtract(y, pred_y)
    exp_rmse = tf.square(diff)
    exp_rmse = tf.cast(exp_rmse, dtype=tf.float32)

    return tf.sqrt(tf.reduce_mean(exp_rmse))
