import tensorflow as tf

import InputHandler as inputHandler

class Model(object):
    def __init__(self,
        data_dir,
        batch_size,
        epochs
            ):
        self.__data_dir__ = data_dir
        self.__batch_size__ = batch_size
        self.__epochs__ = epochs

    def train(self):
        logits, labels = self._inference()
        with tf.name_scope("training"):
            Xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels
            )
            loss = tf.reduce_mean(Xentropy)
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()

        filename_queue = inputHandler.get_filenames_queue(
            data_dir=self.__data_dir__,
            epochs=self.__epochs__,
            is_train=True)
        image_batch_op, label_batch_op = inputHandler.get_data_batch(
                                            filename_queue,
                                            batch_size=self.__batch_size__,
                                            is_train=True)

        with tf.Session() as sess:
            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            while not coord.should_stop():
                image_batch, label_batch = sess.run([image_batch_op, label_batch_op])
                sess.run(train_op,
                    feed_dict={
                        X: image_batch,
                        y: label_batch
                    }
                )
            coord.request_stop()
            coord.join(threads)

    def eval(self):
        logits, labels = self._inference()
        with tf.name_scope("eval"):
            correct = tf.nn.in_top_k(logits, labels, 5)
            accuracy = tf.reduce_mean(tf.cast(
                correct,
                tf.float32
            ))

        filename_queue = inputHandler.get_filenames_queue(
                                            data_dir=self.__data_dir__,
                                            is_train=False)
        image_batch_op, label_batch_op = inputHandler.get_data_batch(
                                                filename_queue,
                                                batch_size=self.__batch_size__,
                                                is_train=False)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            while not coord.should_stop():
                image_batch, label_batch = sess.run([image_batch_op, label_batch_op])
                sess.run(train_op,
                    feed_dict={
                        X: image_batch,
                        y: label_batch
                    }
                )
            coord.request_stop()
            coord.join(threads)

    def _inference(self):
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
        return logits, y

    def save(self):
        with tf.name_scope("init_and_save"):
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()

        return saver
