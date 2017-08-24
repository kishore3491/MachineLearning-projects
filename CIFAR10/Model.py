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
    with tf.name_scope("convolutions"):
        conv1 = tf.layers.conv2d(
            inputs=X,
            filters=32,
            kernel_size=[3,3],
            strides=[1,1],
            padding="SAME",
            activation=tf.nn.relu
        )

        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=64,
            kernel_size=[3,3],
            strides=[2,2],
            padding="SAME",
            activation=tf.nn.relu
        )

        # Pooling layer 2
        pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=[2,2],
            strides=2
        )

        dropout2 = tf.layers.dropout(
            inputs=pool2,
            rate=0.2
        )

        conv3 = tf.layers.conv2d(
            inputs=conv1,
            filters=64,
            kernel_size=[3,3],
            strides=[2,2],
            padding="SAME",
            activation=tf.nn.relu
        )

        # Pooling layer 3
        pool3 = tf.layers.max_pooling2d(
            inputs=conv3,
            pool_size=[2,2],
            strides=2
        )

        dropout3 = tf.layers.dropout(
            inputs=pool3,
            rate=0.2
        )

        # print(dropout3.shape)
    #... more conv layers as needed

    # To flatten, get prev layer's shape, and convert to new shape [None, x]
    with tf.name_scope("flatten"):
        flatten = tf.reshape(
            dropout3,
            shape=[-1, 8*8*64]
        )

    with tf.name_scope("fully_connected"):
        fc2 = tf.layers.dense(
            inputs=flatten,
            units=512,
            activation=tf.nn.relu
        )
        dropout3 = tf.layers.dropout(
            inputs=fc2,
            rate=0.3
        )
        logits = tf.layers.dense(
            inputs=dropout3,
            units=10,
            activation=tf.nn.relu
        )
    return logits
