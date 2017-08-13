# Build a CNN Network

import tensorflow as tf


IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNELS = 3


def inference():
    with tf.name_scope("inputs"):
        X = tf.placeholder(
            tf.float32,
            shape=[None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS],
            name="X"
        )

        y = tf.placeholder(
            tf.int32,
            shape=[None],
            name="y"
        )

    with tf.name_scope("conv1"):
        conv1 = tf.layers.conv2d(
            inputs=X,
            filters=3,
            kernel_size=[3,3],
            strides=[2,2],
            padding="SAME",
            activation=tf.nn.relu
        )

        # Pooling layer 1
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2,2],
            strides=2
        )

    #... more conv layers as needed

    with tf.name_scope("flatten"):
        flatten = tf.reshape(
            pool1,
            shape=[-1, 7*7*3]
        )

    with tf.name_scope("fully_connected"):
        fc = tf.layers.dense(
            inputs=flatten,
            units=1024,
            activation=tf.nn.relu
        )
        logits = tf.layers.dense(
            inputs=fc,
            units=10,
            activation=tf.nn.relu
        )
        Y_pred = tf.nn.softmax(inputs=logits)
    return logits

def train():
    with tf.name_scope("training"):
        Xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=y
        )
        loss = tf.reduce_mean(Xentropy)
        optimizer = tf.train.AdamOptimize()
        training_op = optimizer.minimize(loss)
    return training_op

def eval():
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(
            correct,
            tf.float32
        ))
    return accuracy

def save():
    with tf.name_scope("init_and_save"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

def app_main():
    n_epochs = 1
    batch_size = 100


    # TODO do this before first layer init
    # tf.reset_default_graph()

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for batch in get_next_batch():
                sess.run(
                    training_op,
                    feed_dict={
                        X: batch.X,
                        y: batch.y
                    }
                )

        # TODO
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

        save_path = saver.save(sess, "./my_mnist_model")
