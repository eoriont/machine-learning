from node import Node

class Graph:
    def __init__(self, edges, vertices=None):
        self.nodes = [Node(i, value=v) for i, v in enumerate(vertices)]
        for x, y in edges:
            self.nodes[x].set_neighbor(self.nodes[y])

    def depth_first_search(self, starting_index):
        return self.nodes[starting_index].depth_first_search([])

    def breadth_first_search(self, starting_index):
        queue, result = [self.nodes[starting_index]], []
        while len(queue) > 0:
            next_val = queue.pop()
            queue[:0] = [
                node for node in next_val.neighbors if node not in queue+result]
            result.append(next_val)
        return [node.index for node in result]

    def calc_distance(self, p1, p2):
        gens, current_gen = 0, [p1]
        while p2 not in current_gen:
            current_gen = list(
                {node.index for i in current_gen for node in self.nodes[i].neighbors})
            gens += 1
        return gens

    def calc_shortest_path(self, p1, p2):
        layers = [[self.nodes[p1]]]
        self.nodes[p1].set_parent(None)
        current_layer = 0
        while p2 not in [x.index for l in layers for x in l]:
            new_layer = []
            for node in layers[current_layer]:
                for neighbor in node.neighbors:
                    if neighbor.index not in [x.index for l in layers for x in l]:
                        neighbor.set_parent(node)
                        new_layer.append(neighbor)
            layers.append(new_layer)
            current_layer += 1
        return self.get_path_recursive(self.nodes[p2])

    def get_path_recursive(self, node, path=None):
        path = [] if path is None else path
        return self.get_path_recursive(node.parent, [node.index]+path) if node.parent is not None else [node.index]+path
