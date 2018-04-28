from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
import tensorflow as tf
import cv2 as cv

FLAGS = tf.app.flags.FLAGS
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

def optimize(num_iterations, session, optimizer, data, lr):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        if i % 1000 == 0:
            lr = 0.5*lr

        feed_dict_train = {x_image: x_batch.reshape(train_batch_size,
                                                    28, 28, 1),
                           y_true: y_true_batch,
                           learning_rate: lr}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations


def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    test_batch_size = 256

    # Number of images in the test-set.
    num_test = len(data.test.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x_image: images.reshape(j-i, 28, 28, 1),
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = np.argmax(data.test.labels, axis=1)
    print(cls_true)
    print(cls_pred)
    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)


if __name__ == "__main__":
    # We know that MNIST images are 28 pixels in each dimension.
    img_size = 28

    # Images are stored in one-dimensional arrays of this length.
    img_size_flat = img_size * img_size

    # Tuple with height and width of images used to reshape arrays.
    img_shape = (img_size, img_size)

    # Number of colour channels for the images: 1 channel for gray-scale.
    num_channels = 1

    # Number of classes, one class for each of 10 digits.
    num_classes = 10
    data = input_data.read_data_sets('/tmp/mnist', one_hot=True)
    print("Size of:")
    print("- Training-set:\t\t{}".format(len(data.train.labels)))
    print("- Test-set:\t\t{}".format(len(data.test.labels)))
    print("- Validation-set:\t{}".format(len(data.validation.labels)))

    data.test.cls = np.argmax(data.test.labels, axis=1)
    # Get the first images from the test-set.
    images = data.test.images[0:9]
    # Get the true classes for those images.

    x_image = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, axis=1)


    net = x_image
    net = tf.layers.conv2d(inputs=net, name='layer_conv1', padding='same',
    filters=16, kernel_size=5, activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
    net = tf.layers.conv2d(inputs=net, name='layer_conv2', padding='same',
                       filters=36, kernel_size=5, activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
    net = tf.reshape(net, [-1, np.prod(net.get_shape().as_list()[1:])])
    #net = tf.layers.flatten(net)

    net = tf.layers.dense(inputs=net, name='layer_fc1',
    units=128, activation=tf.nn.relu)

    net = tf.layers.dense(inputs=net, name='layer_fc_out',
                      units=num_classes, activation=None)
    logits = net
    y_pred = tf.nn.softmax(logits=logits, name="y_pred")

    y_pred_cls = tf.argmax(y_pred, axis=1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
    regularizer = tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('layer_fc_out/kernel:0'))

    regularizer += tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('layer_fc1/kernel:0'))

    loss = tf.reduce_mean(cross_entropy) + regularizer
    lr = 1.0
    learning_rate = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss)

    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    train_batch_size = 64
    session = tf.Session(config =tf.ConfigProto(
        device_count = {'GPU': 0}
    ))
    session.run(tf.global_variables_initializer())

    # Counter for total number of iterations performed so far.
    total_iterations = 0
    optimize(8000, session, optimizer, data, lr)

    print_test_accuracy()
    saver = tf.train.Saver()
    saver.save(session, './digitNet')

    session.close()
