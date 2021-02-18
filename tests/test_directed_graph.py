import sys
sys.path.append("src/graphs")
from directed_graph import DirectedGraph
from otest import do_assert

edges = [(0, 1), (1, 2), (3, 1), (4, 3), (1, 4), (4, 5), (3, 6)]
directed_graph = DirectedGraph(edges)

do_assert("graph node children", [[child.index for child in node.children]
                                  for node in directed_graph.nodes], [[1], [2, 4], [], [1, 6], [3, 5], [], []])

do_assert("graph node parents", [[parent.index for parent in node.parents]
                                 for node in directed_graph.nodes], [[], [0, 3], [1], [4], [1], [4], [3]])

do_assert("calc distance 1", directed_graph.calc_distance(0, 3), 3)

do_assert("calc distance 2", directed_graph.calc_distance(3, 5), 3)

do_assert("calc distance 3", directed_graph.calc_distance(0, 5), 3)

do_assert("calc distance 4", directed_graph.calc_distance(4, 1), 2)

do_assert("calc distance 5", directed_graph.calc_distance(2, 4), False)

do_assert("shortest path 1",
          directed_graph.calc_shortest_path(0, 3), [0, 1, 4, 3])
do_assert("shortest path 2",
          directed_graph.calc_shortest_path(3, 5), [3, 1, 4, 5])
do_assert("shortest path 3",
          directed_graph.calc_shortest_path(0, 5), [0, 1, 4, 5])
do_assert("shortest path 4",
          directed_graph.calc_shortest_path(4, 1), [4, 3, 1])
do_assert("shortest path 5", directed_graph.calc_shortest_path(2, 4), False)
