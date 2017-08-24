import argparse
import sys
import math
import tensorflow as tf
import numpy as np

import Model as model
import InputHandler as inputHandler

FLAGS = None

def evaluate():
    """
    Evaluate a previously trained model.
    Steps:
    1. Build model
    2. Create filename_queue with eval data.
    3. Data feed ops
    4. Restore model variables from checkpoint, or Exit.
    5. Run accuracy op against eval data and print accuracy.
    """
    # 1. Build eval model.
    with tf.name_scope("eval"):
        X,y = model.placeholders()
        logits = model.inference(X)
        top_k_op = tf.nn.in_top_k(logits, y, 1)

    # 2. filename_queue with eval data
    filename_queue = inputHandler.get_filenames_queue(
                                        data_dir=FLAGS.data_dir,
                                        is_train=False)
    # 3. Data feed ops, placed on CPU.
    with tf.device('/cpu:0'):
        image_batch_op, label_batch_op = inputHandler.get_data_batch(
                                            filename_queue,
                                            batch_size=FLAGS.batch_size,
                                            is_train=False)

    # merged = tf.summary.merge_all()
    # test_writer = tf.summary.FileWriter(FLAGS.log_dir, tf.get_default_graph())

    with tf.Session() as sess:
        """
        Note: Do not init variables again; This will load new set of variables,
        as if a new training session is being started, ignoring pre-trained
        weights. Just load variables from checkpoint.
        The precision value should be same for different runs of this experiment.
        i.e re-running eval with same checkpoint, data and other values should
        give same precision value. If a different precision value is returned
        for each run, most possibly the pre-trained weights are messed up
        somewhere.
        """
        print(sess.run(tf.report_uninitialized_variables()))
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
            return
        print(sess.run(tf.report_uninitialized_variables()))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        num_iters = int(math.ceil(inputHandler.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL/FLAGS.batch_size))
        i = 0
        true_count = 0

        try:
            while (not coord.should_stop()) and (i < num_iters):
                image_batch, label_batch = sess.run([image_batch_op, label_batch_op])
                predictions = sess.run(top_k_op,
                    feed_dict={
                        X: image_batch,
                        y: label_batch
                    }
                )
                i += 1
                # test_writer.add_summary(summary, i)
                true_count += np.sum(predictions)

        except tf.errors.OutOfRangeError:
            print('Epoch limit reached.')

        precision = true_count / inputHandler.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
        print("Precision: ", precision)
        # test_writer.close()
        coord.request_stop()
        coord.join(threads)
        sess.close()


def main(_):
    # Clear graph
    tf.reset_default_graph()
    # Delete existing directories
    inputHandler.delete_directories(FLAGS.log_dir)
    evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--ckpt_dir", default="/home/kbanala/Work/DataScience/Projects-ML-DL/CIFAR10/checkpoints")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/kbanala/Work/DataScience/DataSets/CIFAR10/cifar-10-batches-bin",
        help='directory for datasets.')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='/tmp/tf/CIFAR10/eval',
        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
