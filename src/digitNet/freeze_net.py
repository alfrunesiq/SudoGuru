import sys

import tensorflow as tf
from tensorflow.python.framework import graph_util
tf.reset_default_graph()

if len(sys.argv) > 1:
    file_base = sys.argv[1]
else:
    file_base = "./digitNet"

saver = tf.train.import_meta_graph(file_base + ".meta", clear_devices=True)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
sess = tf.Session(config =tf.ConfigProto(
        device_count = {"GPU": 0})
)
saver.restore(sess, file_base)

output_node_names="y_pred"
output_graph_def = graph_util.convert_variables_to_constants(
            sess, # The session
            input_graph_def, # input_graph_def is useful for retrieving the nodes
            output_node_names.split(",")
)


output_graph= file_base + ".pb"
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
sess.close()
