
import tensorflow as tf

def softmax_accuracy(preds, labels):
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.to_int64(labels))
    accuracy_all = tf.cast(correct_prediction, tf.float32)

    return tf.reduce_mean(accuracy_all)
