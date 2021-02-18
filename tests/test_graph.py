import sys
sys.path.append("src/graphs")
from graph import Graph
from otest import do_assert

edges = [(0, 1), (1, 2), (1, 3), (3, 4), (1, 4)]
vertices = ['a', 'b', 'c', 'd', 'e']
graph = Graph(edges, vertices)
do_assert("graph build from edges", [
        node.index for node in graph.nodes], [0, 1, 2, 3, 4])

do_assert("node values", [node.value for node in graph.nodes], [
        'a', 'b', 'c', 'd', 'e'])

do_assert("node neighbors", [len(node.neighbors)
                            for node in graph.nodes], [1, 4, 1, 2, 2])

edges = [(0, 1), (1, 2), (1, 3), (3, 4), (1, 4), (4, 5)]
vertices = [0, 1, 2, 3, 4, 5]
graph = Graph(edges, vertices)

do_assert("graph shortest path #1", graph.calc_shortest_path(0, 4), [0, 1, 4])

do_assert("graph shortest path #2",
          graph.calc_shortest_path(5, 2), [5, 4, 1, 2])

do_assert("graph shortest path #3",
          graph.calc_shortest_path(0, 5), [0, 1, 4, 5])

do_assert("graph shortest path #4", graph.calc_shortest_path(4, 1), [4, 1])

do_assert("graph shortest path #5", graph.calc_shortest_path(3, 3), [3])

do_assert("graph shortest distance #1", graph.calc_distance(0, 4), 2)

do_assert("graph shortest distance #2", graph.calc_distance(5, 2), 3)

do_assert("graph shortest distance #3", graph.calc_distance(0, 5), 3)

do_assert("graph shortest distance #4", graph.calc_distance(4, 1), 1)

do_assert("graph shortest distance #5", graph.calc_distance(3, 3), 0)
