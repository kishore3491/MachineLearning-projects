import argparse
import sys
import tensorflow as tf
from Model import Model

FLAGS = None

def main(_):
    # Clear graph
    tf.reset_default_graph()
    # Delete log directories, if any.
    if tf.gfile.Exists(FLAGS.log_dir):
      tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    model = Model(
                data_dir=FLAGS.data_dir,
                batch_size=FLAGS.batch_size,
                epochs=FLAGS.epochs
            )
    model.train()
    model.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_dir", default="/tmp/CIFAR10_train")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/kbanala/Work/DataScience/DataSets/CIFAR10",
        help='directory for datasets.')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='/tmp/tf/CIFAR10',
        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
