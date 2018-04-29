import os
import sys

import numpy as np
import tensorflow as tf
import cv2 as cv
from tensorflow.examples.tutorials.mnist import input_data

num_epochs = 500
batch_size = 2096
lr = 1.0


img_size     = 32
num_channels = 1
num_classes  = 9   # 0-9 in this case


x_train = np.load("/tmp/English/digits_img.npy")
y_train_cls = np.load("/tmp/English/digits_lbl.npy")
# Translate to one-hot
y_train_true = np.zeros((x_train.shape[0], 9), dtype=np.float)
y_train_true[np.arange(x_train.shape[0]),y_train_cls-1] = 1.0

if (len(sys.argv) > 1 and sys.argv[1] == "mnist"):
    data = input_data.read_data_sets("/tmp/mnist/")
    ## Train on it ALL!!!!
    for data_set in [data.train, data.test, data.validation]:
        mnist_train = data_set.images[data_set.labels != 0].reshape(-1, 28, 28)
        mnist_train_ext = np.zeros((mnist_train.shape[0], 32, 32), dtype=np.float32)
        mnist_train_ext[:,2:30, 2:30] = mnist_train
        mnist_train_one_hot = np.zeros((mnist_train.shape[0], num_classes), dtype=np.float32)
        mnist_train_one_hot[np.arange(mnist_train.shape[0]), data_set.labels[data_set.labels!=0]-1] = 1.0
        y_train_cls = np.concatenate([y_train_cls, data_set.labels[data_set.labels != 0]], axis=0)
        x_train = np.concatenate([x_train, mnist_train_ext], axis=0)
        y_train_true = np.concatenate([y_train_true, mnist_train_one_hot], axis=0)

print("Training set size: %d" % x_train.shape[0])
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

net1    = tf.layers.conv2d(inputs=x, name="layer_conv1", padding="valid",
                           filters=24, kernel_size=5, activation=tf.nn.relu)
print(net1.get_shape().as_list())
net1p   = tf.layers.max_pooling2d(inputs=net1, name="layer_pool1", pool_size=2, strides=2)
net2    = tf.layers.conv2d(inputs=net1p, name="layer_conv2", padding="valid",
                           filters=36, kernel_size=5, activation=tf.nn.relu)
print(net2.get_shape().as_list())
net2p   = tf.layers.max_pooling2d(inputs=net2, name="layer_pool2", pool_size=2, strides=2)
net3    = tf.layers.conv2d(inputs=net2p, name="layer_conv3", padding="valid",
                           filters=256, kernel_size=5, activation=tf.nn.relu)
print(net3.get_shape().as_list())
flatten = tf.reshape(net3, [-1, np.prod(net3.get_shape().as_list()[1:])])
fc1     = tf.layers.dense(inputs=flatten, name="layer_fc1", units=128, activation=tf.nn.relu)
logits  = tf.layers.dense(inputs=fc1, name="layer_fc2", units=num_classes, activation=None)

y_pred     = tf.nn.softmax(logits=logits, name="y_pred")
y_pred_cls = tf.argmax(y_pred, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
regularizer   = tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('layer_fc1/kernel:0'))
regularizer  += tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('layer_fc2/kernel:0'))
regularizer  += 0.1*tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('layer_conv1/kernel:0'))
regularizer  += 0.1*tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('layer_conv2/kernel:0'))
regularizer  += 0.1*tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('layer_conv3/kernel:0'))




loss = tf.reduce_mean(cross_entropy) + regularizer
learning_rate = tf.placeholder(tf.float32, shape=[])
optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred_cls, y_true_cls), tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())

## Train network:
num_samples = x_train.shape[0]
a_range = np.arange(num_samples)

print(x_train.shape)
for i in range(num_epochs):
    if i % 30 == 29:
        lr /= 2.0

    if i == 150:
        lr = 1.0

    j = 0
    a_range = np.random.choice(a_range, num_samples, replace=False)
    while(j <= num_samples):
        # Draw a random sample from batch
        k = min(j+batch_size, num_samples)
        x_batch = x_train[a_range[j:k],:,:]
        y_true_batch = y_train_true[a_range[j:k],:]

        # inject noise (regularization)
        if i > 200:
            x_batch = np.clip((x_batch + 0.01*np.random.randn(*x_batch.shape)), 0.0, 1.0)

        # Create feeddict
        feed_dict_train = {x: x_batch.reshape((-1, 32, 32, 1)),
                           y_true: y_true_batch,
                           learning_rate: lr}
        session.run(optimizer, feed_dict=feed_dict_train)
        j += batch_size

    # Report accuracy
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    msg = "Iteration: %04d | Training Accuracy: %02.2f" % (i, acc*100)
    print(msg)


saver = tf.train.Saver()
saver.save(session, './digitNet')

session.close()
