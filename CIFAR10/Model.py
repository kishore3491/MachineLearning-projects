import os
import tensorflow as tf

import InputHandler as inputHandler

class Model(object):
    def __init__(self,
        data_dir,
        log_dir,
        batch_size,
        epochs
            ):
        self.__data_dir__ = data_dir
        self.__log_dir__ = log_dir
        self.__batch_size__ = batch_size
        self.__epochs__ = epochs
        self.graph = tf.Graph()
        self.saver = None

    def train(self):
        with self.graph.as_default():
            with tf.name_scope("training"):
                X,y = self._placeholders()
                logits = self._inference(X)
                Xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits,
                    labels=y
                )
                loss = tf.reduce_mean(Xentropy)

                # Add this to TensorBoard
                tf.summary.scalar('loss', loss)

                optimizer = tf.train.AdamOptimizer()
                train_op = optimizer.minimize(loss)

            filename_queue = inputHandler.get_filenames_queue(
                data_dir=self.__data_dir__,
                epochs=self.__epochs__,
                is_train=True)

            with tf.device('/cpu:0'):
                image_batch_op, label_batch_op = inputHandler.get_data_batch(
                                                filename_queue,
                                                batch_size=self.__batch_size__,
                                                is_train=True)

            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.__log_dir__ + '/train', tf.get_default_graph())

            init = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
            with tf.Session(graph=self.graph) as sess:
                sess.run(init)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
                try:
                    while not coord.should_stop():
                        image_batch, label_batch = sess.run([image_batch_op, label_batch_op])
                        summary, _ = sess.run([merged, train_op],
                            feed_dict={
                                X: image_batch,
                                y: label_batch
                            }
                        )
                        train_writer.add_summary(summary)
                        # print("Loss: " + loss.eval())
                except tf.errors.OutOfRangeError:
                    print('Done training -- epoch limit reached')
                train_writer.close()
                coord.request_stop()
                coord.join(threads)

                checkpoint_file = os.path.join(self.__log_dir__, 'model.ckpt')
                saver = tf.train.Saver()
                saver.save(sess, checkpoint_file)
                self.saver = saver

    def eval(self):
        if self.saver is None:
            print('No saver found, ensure model is trained before running eval.')
            return
        with tf.Session(graph=self.graph) as sess:
            ckpt = tf.train.get_checkpoint_state(self.__log_dir__)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')
                return

            with tf.name_scope("eval"):
                X,y = self._placeholders()
                logits = self._inference(X)
                correct = tf.nn.in_top_k(logits, y, 5)
                accuracy = tf.reduce_mean(tf.cast(
                    correct,
                    tf.float32
                ))
                # Add this to TensorBoard
                tf.summary.scalar('accuracy', accuracy)

            filename_queue = inputHandler.get_filenames_queue(
                                                data_dir=self.__data_dir__,
                                                is_train=False)
            with tf.device('/cpu:0'):
                image_batch_op, label_batch_op = inputHandler.get_data_batch(
                                                    filename_queue,
                                                    batch_size=self.__batch_size__,
                                                    is_train=False)

            merged = tf.summary.merge_all()
            test_writer = tf.summary.FileWriter(self.__log_dir__ + '/test', tf.get_default_graph())

            local_init = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
            sess.run(local_init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                while not coord.should_stop():
                    image_batch, label_batch = sess.run([image_batch_op, label_batch_op])
                    summary, _ = sess.run([merged, accuracy],
                        feed_dict={
                            X: image_batch,
                            y: label_batch
                        }
                    )
                    test_writer.add_summary(summary)
                    # print("Loss: " + loss.eval())
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            test_writer.close()
            coord.request_stop()
            coord.join(threads)

    def _placeholders(self):
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

    def _inference(self, X):
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
