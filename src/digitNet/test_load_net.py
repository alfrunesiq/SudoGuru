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
print("Input tensor:")
print(a)
print("Output tensor:")
print(b)

for i, node in enumerate(graph_def.node):
    print("%d %s %s" % (i, node.name, node.op))
    [print(u'└─── %d ─ %s' % (i, n)) for i, n in enumerate(node.input)]
