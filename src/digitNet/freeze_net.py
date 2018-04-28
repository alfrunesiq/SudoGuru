import tensorflow as tf
from tensorflow.python.framework import graph_util
tf.reset_default_graph()

saver = tf.train.import_meta_graph('./digitNet.meta', clear_devices=True)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
sess = tf.Session(config =tf.ConfigProto(
        device_count = {'GPU': 0}))
saver.restore(sess, "./digitNet")

output_node_names="y_pred"
output_graph_def = graph_util.convert_variables_to_constants(
            sess, # The session
            input_graph_def, # input_graph_def is useful for retrieving the nodes 
            output_node_names.split(",")
)


output_graph="./digitNet.pb"
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
sess.close()
