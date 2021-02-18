import sys
sys.path.append("src/graphs")
from node import Node
from otest import do_assert


string_node = Node(0)
do_assert("Initialize value", string_node.index, 0)

string_node.set_value('asdf')
do_assert("Node set_value", string_node.value, 'asdf')

do_assert("Initialize neighbors", string_node.neighbors, [])

numeric_node = Node(1)
numeric_node.set_value(765)
numeric_node.set_neighbor(string_node)
do_assert("Node set_neighbor", numeric_node.neighbors[0].value, 'asdf')
do_assert("Node set_neighbor from another node",
          string_node.neighbors[0].value, 765)

array_node = Node(2)
array_node.set_value([[1, 2], [3, 4]])
array_node.set_neighbor(numeric_node)
do_assert("Set third node neighbor", array_node.neighbors[0].value, 765)
do_assert("Set third node neighbor from other node",
          numeric_node.neighbors[1].value, [[1, 2], [3, 4]])
