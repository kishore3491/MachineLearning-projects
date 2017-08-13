import tensorflow as tf

import InputHandler as inputHandler

class Model(object):
    def init(self):
        pass

    def train(self):
        logits = self._inference()
        with tf.name_scope("training"):
            Xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=y
            )
            loss = tf.reduce_mean(Xentropy)
            optimizer = tf.train.AdamOptimize()
            train_op = optimizer.minimize(loss)

        # TODO Get data_dir & batch_size from Flags.
        filename_queue = inputHandler.get_filenames_queue(data_dir=None)
        image_batch_op, label_batch_op = inputHandler.get_data_batch(filename_queue, batch_size=None)
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            while not coord.should_stop():
                image_batch, label_batch = sess.run([image_batch_op, label_batch_op])
                sess.run(train_op,
                    feed_dict: {
                        X: image_batch,
                        y: label_batch
                    }
                )
            coord.request_stop()
            coord.join(threads)

    def eval(self):
        pass

    def _inference(self):
        pass
