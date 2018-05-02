"""
This is a new iteration of freeze_net.py that
also removes dropout layers from the graphs
and removes unneeded placeholders
"""
import sys
import re

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

## Remove dropout layer and redundant placehoders
## (i.e. Placeholders without custom names)
connections = {}
del_list    = []
dropout_detected = False
for (i, node) in enumerate(output_graph_def.node):
    if re.search("dropout", node.name) != None:
        if not dropout_detected:
            print("Removing Dropout layers...")
        dropout_detected = True
        if node.op == "Shape":
            layername = node.name.split("/")[0]
            if not layername in connections.keys():
                connections.update({layername: {"prew": i}})
            else:
                connections[layername].update({"prew": i})
        del_list.append(node.name)

    elif len(node.input) > 0 and \
         "dropout" in node.input[0]:
        layername = node.input[0].split("/")[0]
        if not layername in connections.keys():
            connections.update({layername: {"next": i}})
        else:
            connections[layername].update({"next": i})
    elif node.name.startswith("Placeholder"):
        del_list.append(node.name)


for connection in connections.values():
    print("%d" % connection["prew"], "->", "%d" % connection["next"])
    output_graph_def.node[connection["next"]].input[0] =\
                    output_graph_def.node[connection["prew"]].input[0]

deleted = True
while(deleted):
    deleted=False
    for (i, node) in enumerate(output_graph_def.node):
        if node.name in del_list:
            print ("Deleting: %s" % output_graph_def.node[i].name)
            del output_graph_def.node[i]
            deleted = True

for i, node in enumerate(output_graph_def.node):
    print("%d %s %s" % (i, node.name, node.op))
    [print(u'└─── %d ─ %s' % (i, n)) for i, n in enumerate(node.input)]

output_graph= file_base + ".pb"
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
sess.close()
