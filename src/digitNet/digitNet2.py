import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

num_iterations = 10000
batch_size     = 64

img_size     = 32
num_channels = 1
num_classes  = 9   # 1-9 in this case

x_train = np.load("/tmp/digits_img.npy")
y_train_cls = np.load("/tmp/digits_lbl.npy")
# Translate to one-hot
y_train_true = np.zeros((x_train.shape[0], 9), dtype=np.float)
y_train_true[np.arange(x_train.shape[0]),y_train_cls-1] = 1.0

print("Training set size: %d", y_train_true.shape[0])

x = tf.placeholder(tf.float32, shape = [None, img_size, img_size, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

net1    = tf.layers.conv2d(inputs=x, name="layer_conv1", padding="same",
                           filters = 16, kernel_size=5, activation=tf.nn.relu)
net1p   = tf.layers.max_pooling2d(inputs=net1, name="layer_pool1", pool_size=2, strides=2)
net2    = tf.layers.conv2d(inputs=net1p, name="layer_conv2", padding="same",
                           filters=36, kernel_size=5, activation=tf.nn.relu)
net2p   = tf.layers.max_pooling2d(inputs=net2, name="layer_pool2", pool_size=2, strides=2)
net3    = tf.layers.conv2d(inputs=net1p, name="layer_conv3", padding="same",
                           filters=48, kernel_size=5, activation=tf.nn.relu)
net3p   = tf.layers.max_pooling2d(inputs=net2, name="layer_pool3", pool_size=2, strides=2)

flatten = tf.reshape(net3p, [-1, np.prod(net2p.get_shape().as_list()[1:])])
fc1     = tf.layers.dense(inputs=flatten, name="layer_fc1", units=128, activation=tf.nn.relu)
logits  = tf.layers.dense(inputs=fc1, name="layer_fc2", units=num_classes, activation=None)

y_pred     = tf.nn.softmax(logits=logits, name="y_pred")
y_pred_cls = tf.argmax(y_pred, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
regularizer   = tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('layer_fc1/kernel:0'))
regularizer  += tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('layer_fc2/kernel:0'))


loss = tf.reduce_mean(cross_entropy) + regularizer
lr = 0.01
learning_rate = tf.placeholder(tf.float32, shape=[])
optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred_cls, y_true_cls), tf.float32))

session = tf.Session(config =tf.ConfigProto(
    device_count = {'GPU': 0}
))
session.run(tf.global_variables_initializer())

## Train network:
num_samples = x_train.shape[0]
a_range = np.arange(num_samples)
for i in range(num_iterations):
    if i % 3000 == 2999:
        lr /= 2.0
        batch_size += 64
    # Draw a random sample from batch
    indecies = np.random.choice(a_range, batch_size, replace=False)
    x_batch = x_train[indecies,:,:]
    y_true_batch = y_train_true[indecies,:]

    # inject noise (regularization)
    x_batch = np.clip((x_batch + 0.05*np.random.randn(*x_batch.shape)), 0.0, 1.0)

    # Create feeddict
    feed_dict_train = {x: x_batch.reshape(batch_size, img_size,
                                          img_size, num_channels),
                       y_true: y_true_batch,
                       learning_rate: lr}

    session.run(optimizer, feed_dict=feed_dict_train)

    if i % 100 == 0:
        # Report accuracy
        acc = session.run(accuracy, feed_dict=feed_dict_train)
        msg = "Iteration: %04d | Training Accuracy: %02.2f" % (i, acc*100)
        print(msg)


saver = tf.train.Saver()
saver.save(session, './digitNet')

session.close()
