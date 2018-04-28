import tensorflow as tf

with tf.gfile.GFile('./digitNet.pb', "rb") as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(
        graph_def,
        input_map=None,
        return_elements=None,
        name=""
        )

a, b= tf.import_graph_def(graph_def,
                          return_elements=['x',
                                           'y_pred'],
                          name='')

print(a)
