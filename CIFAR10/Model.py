import tensorflow as tf
import InputHandler as inputHandler


def placeholders():
    with tf.name_scope("inputs"):
        X = tf.placeholder(
            tf.float32,
            shape=[
                    None,
                    inputHandler.IMG_HEIGHT,
                    inputHandler.IMG_WIDTH,
                    inputHandler.IMG_CHANNELS
                ],
            name="X"
        )
        y = tf.placeholder(
            tf.int32,
            shape=[None],
            name="y"
        )
    return X,y

def inference(X):
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

    # To flatten, get prev layer's shape, and convert to new shape [None, x]
    with tf.name_scope("flatten"):
        flatten = tf.reshape(
            pool1,
            shape=[-1, 8*8*3]
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
    return logits
