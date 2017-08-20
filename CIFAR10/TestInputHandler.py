import tensorflow as tf
import InputHandler as inputHandler

with tf.device('/cpu:0'):

    # Read a record, getting filenames from filename_queue.
    reader = tf.FixedLengthRecordReader(
        record_bytes=3073
    )
    filename_queue = inputHandler.get_filenames_queue(
                                        data_dir='/home/kbanala/Work/DataScience/DataSets/CIFAR10/cifar-10-batches-bin',
                                        is_train=False,
                                        epochs=1)
    image_batch_op, label_batch_op = inputHandler.get_data_batch(
                                        filename_queue,
                                        batch_size=10,
                                        is_train=False,
                                        num_process_threads=1)
    labels_max_op = tf.reduce_max(label_batch_op)
    # Or test against the raw data directly.
    # key, value = reader.read(filename_queue)
    init = tf.local_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(tf.report_uninitialized_variables()))
        # create a coordinator, launch the queue runner threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            for step in range(100): # do some iterations
                if coord.should_stop():
                    break
                labels_max = sess.run(labels_max_op)
                # k,v = sess.run([key,value])
                assert(labels_max < 10)
                # TODO Try loading some images to opencv or similar.
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        coord.request_stop()
        coord.join(threads)
