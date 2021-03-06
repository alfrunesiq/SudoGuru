import os
import sys

import numpy as np
import tensorflow as tf
import cv2 as cv
from tensorflow.examples.tutorials.mnist import input_data

"""
This is a vanilla minimal convolutional network similar to LeNet
Below I train the classifier using AdadeltaOptimizer and L2-
regularization and dropout, as well as dataset augmentation to
improove upon generalization.
The trainingset consists of mnist (with zero-images omittet) and
chars78k (with digits 1-9 non-italic fonts) aswell as some images
gathered from actual sudokuboards totalling in 68.128 images.
"""

num_epochs = 1000
batch_size = 4096

#Fixed regularization parameters
lambda_conv1 = 1e-3
lambda_conv2 = 5e-4
lambda_conv3 = 1e-6
lambda_FC    = 1e-4

## Parameters over iterations:
dropout_upd   = [      1,  100,  250,  500, 750, 900, 950, 1750, 1900, num_epochs+1]
dropRate_FC   = [0, 0.75, 0.65,  0.6,  0.5, 0.25,  0.1, 0,   0.1,    0]
dropRate_conv = [0,  0.7,  0.6, 0.55,  0.5, 0.25,  0.1, 0,  0.05,    0]

lr_decay_intrvl = 50
learningRate    = 1.0

## Conv kernel size is fixed to 5 (since it adds up)
conv1_depth   = 6
conv2_depth   = 10
conv3_depth   = 100
fc1_size      = 32

img_size     = 32
num_channels = 1
num_classes  = 9   # 0-9 in this case

dropout_rate_FC = tf.placeholder(tf.float32, [])
dropout_rate_conv = tf.placeholder(tf.float32, [])

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

## Smooth the one-hot labels a little bit
eps = 1e-8
y_train_true[y_train_cls == 6] = np.array([eps, eps, eps, eps, 0, 1-6*eps, 0.5*eps, 0.5*eps, eps])
y_train_true[y_train_cls == 5] = np.array([eps, eps, eps, eps, 1-6*eps, 0, 0.5*eps, 0.5*eps, eps])
y_train_true[y_train_cls == 1] = np.array([1-6.5*eps, eps, eps, 0.5*eps, eps, eps, 0, eps, eps])
y_train_true[y_train_cls == 7] = np.array([0, eps, eps, eps, eps, eps, 1-6.5*eps, eps, 0.5*eps])
y_train_true[y_train_cls == 8] = np.array([eps, eps, 0, eps, eps, 0.5*eps, eps, 1-6*eps, 0.5*eps])
y_train_true[y_train_cls == 3] = np.array([eps, eps, 1-6*eps, eps, eps, 0.5*eps, eps, 0, 0.5*eps])
y_train_true[y_train_cls == 9] = np.array([eps, eps, eps, 0.5*eps, eps, eps, eps, eps, 1-7.5*eps])
y_train_true[y_train_cls == 2] = np.array([eps, 1-6*eps, eps, eps, eps, eps, 0.5*eps, eps, 0.5*eps])
y_train_true[y_train_cls == 4] = np.array([eps, eps, eps, 1-7.5*eps, eps, eps, eps, eps, 0.5*eps])

x_train = np.concatenate([x_train, np.float32(np.random.uniform(0,1,(64,32,32))),\
                          np.zeros((9,32,32),dtype=np.float32),
                          np.ones((1,32,32),dtype=np.float32)], axis=0)
y_train_true = np.concatenate([y_train_true, \
                               1/9*np.ones((x_train.shape[0]-\
                                            y_train_true.shape[0],9))], axis=0)
print("Training set size: %d" % x_train.shape[0])


x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

net1    = tf.layers.conv2d(inputs=x, name="layer_conv1", padding="valid",
                           filters=conv1_depth, kernel_size=5, activation=tf.nn.relu)
net1p   = tf.layers.max_pooling2d(inputs=net1, name="layer_pool1", pool_size=2, strides=2)
dropout_conv1 = tf.layers.dropout(inputs=net1p, name="layer_dropout_conv1", \
                                  training=True, rate=dropout_rate_conv)

net2    = tf.layers.conv2d(inputs=dropout_conv1, name="layer_conv2", padding="valid",
                           filters=conv2_depth, kernel_size=5, activation=tf.nn.relu)
net2p   = tf.layers.max_pooling2d(inputs=net2, name="layer_pool2", pool_size=2, strides=2)
dropout_conv2 = tf.layers.dropout(inputs=net2p, name="layer_dropout_conv2", \
                                  training=True, rate=dropout_rate_conv)

net3    = tf.layers.conv2d(inputs=dropout_conv2, name="layer_conv3", padding="valid",
                           filters=conv3_depth, kernel_size=5, activation=tf.nn.relu)
dropout_conv3 = tf.layers.dropout(inputs=net3, name="layer_dropout_conv3", \
                                  training=True, rate=dropout_rate_conv)

flatten = tf.reshape(dropout_conv3, [-1, np.prod(net3.get_shape().as_list()[1:])])
fc1     = tf.layers.dense(inputs=flatten, name="layer_fc1",
                          units=fc1_size, activation=tf.nn.relu)
dropout_FC = tf.layers.dropout(inputs=fc1, rate=dropout_rate_FC, training=True, name="layer_dropout_FC")
logits  = tf.layers.dense(inputs=dropout_FC, use_bias=False, name="layer_fc2", units=num_classes, activation=None)

y_pred     = tf.nn.softmax(logits=logits, name="y_pred")
y_pred_cls = tf.argmax(y_pred, axis=1)

num_params = 0
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=logits)
regularizer   = lambda_FC*tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('layer_fc1/kernel:0'))
regularizer   += lambda_FC*tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('layer_fc2/kernel:0'))

regularizer  += lambda_conv1*tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('layer_conv1/kernel:0'))
regularizer  += lambda_conv2*tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('layer_conv2/kernel:0'))
regularizer  += lambda_conv3*tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('layer_conv3/kernel:0'))

for layer in ["layer_fc1", "layer_fc2", "layer_conv1", "layer_conv2", "layer_conv3"]:
    weights = tf.get_default_graph().get_tensor_by_name(layer + "/kernel:0")
    num_params += np.prod(weights.get_shape().as_list())
    print ("%s weights:" % layer, weights.get_shape().as_list())
    if layer != "layer_fc2":
        bias = tf.get_default_graph().get_tensor_by_name(layer + "/bias:0")
        num_params += np.prod(bias.get_shape().as_list())
        print ("%s bias:   " % layer, bias.get_shape().as_list())

    if layer in ["layer_conv1", "layer_conv2"]:
        for i in range(weights.get_shape().as_list()[-1]):
            for j in range(i+1, weights.get_shape().as_list()[-1]):
                regularizer += 2e-3/(tf.nn.l2_loss(weights[:,:,:,i] - weights[:,:,:,j])+1)


print("--Total number of parameters:%d" % num_params)




loss = tf.reduce_mean(cross_entropy) + regularizer
learning_rate = tf.placeholder(tf.float32, shape=[])
optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred_cls, y_true_cls), tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())

## Train network:
num_samples = x_train.shape[0]
a_range = np.arange(num_samples)

# Constructed data of "nothing" (added after 1000 epochs)
construct = np.zeros((4,32,32), dtype=np.float32)
construct += np.clip((construct + 0.03*np.random.randn(*construct.shape)), 0.0, 1.0)
construct[:,2:30,2:30] = 1.0


print(x_train.shape)
drop_idx      = 0
drop_next_upd = dropout_upd[0]
drop_FC       = dropRate_FC[0]
drop_conv     = dropRate_conv[0]
for i in range(num_epochs):
    if i == drop_next_upd:
        drop_idx += 1
        drop_next_upd = dropout_upd[drop_idx]
        drop_FC       = dropRate_FC[drop_idx]
        drop_conv     = dropRate_conv[drop_idx]

    if i % lr_decay_intrvl == lr_decay_intrvl-1:
        if learningRate > 0.01:
            learningRate *= 0.9
        elif learningRate > 0.001:
            learningRate *= 0.95
        else:
            learningRate *= 0.99

    if i == 150:
        learningRate = 1.0

    j = 0
    # Shuffle indices
    a_range = np.random.choice(a_range, num_samples, replace=False)
    while(j <= num_samples):
        # Draw a random sample from batch
        k = min(j+batch_size, num_samples)
        x_batch = x_train[a_range[j:k],:,:]
        y_true_batch = y_train_true[a_range[j:k],:]

        # inject noise (regularization)
        if i > 100:
            l = np.int((j+k)/2.0)
            if i > 950:
                x_batch[j:l,:,:] = np.clip((x_batch[j:l,:,:] + \
                                            0.1*np.random.randn(*x_batch[j:l,:,:].shape)), 0.0, 1.0)
            elif i > 700:
                x_batch[j:l,:,:] = np.clip((x_batch[j:l,:,:] + \
                                            0.05*np.random.randn(*x_batch[j:l,:,:].shape)), 0.0, 1.0)
            else:
                x_batch[j:l,:,:] = np.clip((x_batch[j:l,:,:] + \
                                            0.01*np.random.randn(*x_batch[j:l,:,:].shape)), 0.0, 1.0)

            if i == 400:
                ## Add pure noise images and over- and understimulus images, to
                ## promote network to express uncertainty when an erronous cropped
                ## image is classified
                x_train = np.concatenate([x_train, np.float32(np.random.uniform(0,1,(64,32,32))),\
                                          np.zeros((22,32,32),dtype=np.float32),
                                          np.ones((2,32,32),dtype=np.float32),
                                          construct], axis=0)
                y_train_true = np.concatenate([y_train_true, \
                                               1/9*np.ones((x_train.shape[0]-\
                                                            y_train_true.shape[0],9))], axis=0)
                num_samples = x_train.shape[0]
                a_range = np.arange(num_samples)

        # Create feeddict
        feed_dict_train = {x: x_batch.reshape((-1, 32, 32, 1)),
                           y_true: y_true_batch,
                           learning_rate: learningRate,
                           dropout_rate_FC: drop_FC,
                           dropout_rate_conv: drop_conv}
        session.run(optimizer, feed_dict=feed_dict_train)
        j += batch_size
    # end while

    if i % 10 == 0:
        # Report accuracy
        acc = session.run(accuracy, feed_dict=feed_dict_train)
        msg =  "Epoch: %04d | Training Accuracy: %02.2f\n" % (i, acc*100)
        msg += "Dropout rates: [%1.3f, %1.3f]\n" % (drop_conv, drop_FC)
        msg += "Learning rate: %1.3e" % learningRate
        print(msg)


saver = tf.train.Saver()
saver.save(session, './digitNet')

session.close()
