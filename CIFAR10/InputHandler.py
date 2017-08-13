# Goal:
# Read Cifar-10 Binary input files,
# and convert them to TF processable feeds.


import os
from six.moves import xrange
import tensorflow as tf


# CIFAR-10 global constants
NUM_CLASSES = 10
IMAGE_SIZE = 32
IMAGE_CHANNELS = 3
LABEL_BYTES = 1
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

class ImageRecord(object):
    pass


def read_data(filename_queue):
    images_bytes = IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS
    # Compute how many bytes to read per image.
    record_bytes = images_bytes + LABEL_BYTES

    record = ImageRecord()
    record.height = IMAGE_SIZE
    record.width = IMAGE_SIZE
    record.channels = IMAGE_CHANNELS

    # Read a record, getting filenames from filename_queue.
    reader = tf.FixedLengthRecordReader(
        record_bytes=record_bytes
    )
    record.key, value = reader.read(filename_queue)

    # Convert from a string to vector of uint8
    record_data = tf.decode_raw(value, tf.uint8)

    record.label = tf.cast(
        tf.strided_slice(record_data, [0], [LABEL_BYTES]),
        tf.int32
    )

    # The remaining bytes after the label
    # Reshape image from vector to 3D tensor
    depth_major = tf.reshape(
        tf.strided_slice(
            record_data,
            [LABEL_BYTES],
            [record_bytes]
        ),
        [record.channels, record.height, record.width]
    )
    # Convert from [channels, height, width] to [height, width, channels]
    record.uint8image = tf.transpose(
        depth_major,
        [1,2,0]
    )
    return record


def _generate_image_label_batch(image_op, label_op, min_queue_examples, batch_size,
        shuffle):
    # Create a queue that may shuffle examples(?),
    # and then read 'batch_size' images + labels from the example queue.
    num_process_threads = 16
    if shuffle:
        image_batch_op, label_batch_op = tf.train.shuffle_batch(
            [image_op, label_op],
            batch_size=batch_size,
            num_threads=num_process_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples
        )
    else:
        image_batch_op, label_batch_op = tf.train.batch(
            [image_op, label_op],
            batch_size=batch_size,
            num_threads=num_process_threads,
            capacity=min_queue_examples + 3 * batch_size
        )

    # Display training images in the visualizer
    tf.summary.image('images', image_batch_op)

    # Return image and label batch tensors
    return image_batch_op, tf.reshape(label_batch_op, [batch_size])


def get_filenames_queue(is_train=True, data_dir):
    # Step 1: Read filenames from data directory.
    if is_train:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                for i in xrange(1,6)
            ]
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]

    # Step 2: Check if files exits
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError("Failed to find file: " + f)

    # Step 3: Create a queue that produces the filenames to read & return
    return tf.train.string_input_producer(filenames)


def get_data_batch(filename_queue, batch_size):
    # Step 1: Read examples from files in the filename queue.
    read_input = read_data(filename_queue)
    recasted_image = tf.cast(read_input.uint8image, tf.float32)

    # Step 2: Preprocess image
    # resized_image = tf.image.resized_image_with_crop_or_pad(
    #     image=reshaped_image,
    #     target_height=IMAGE_SIZE,
    #     target_width=IMAGE_SIZE
    # )

    # Step 3: Preprocess image 2
    float_image = tf.image.per_image_standardization(recasted_image)

    # Step 4: Set the shapes of tensors.
    float_image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
    read_input.label.set_shape([1])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                            min_fraction_of_examples_in_queue)

    # Step 5: Generate a batch of images and labels by building
    # up a queue of examples.
    return _generate_image_label_batch(
        float_image,
        read_input.label,
        min_queue_examples,
        batch_size,
        shuffle=False
    )
