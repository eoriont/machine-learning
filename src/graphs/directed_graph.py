from directed_node import DirectedNode

class DirectedGraph:
    def __init__(self, edges, vertices=None):
        vertices = self.get_vertices(edges) if vertices is None else vertices
        self.nodes = [DirectedNode(i, value=v) for i, v in enumerate(vertices)]
        for x, y in edges:
            self.nodes[x].add_child(self.nodes[y])
            self.nodes[y].add_parent(self.nodes[x])

    def get_vertices(self, edges):
        v = []
        for x, y in edges:
            if x not in v:
                v.append(x)
            if y not in v:
                v.append(y)

        return sorted(v)

    def calc_distance(self, p1, p2):
        gens, current_gen = 0, [p1]
        last_gen = []
        while p2 not in current_gen:
            current_gen = list(
                {node.index for i in current_gen for node in self.nodes[i].children})
            gens += 1
            last_gen_difference = len(
                set(current_gen).symmetric_difference(set(last_gen)))
            if last_gen_difference == 0:
                return False
            last_gen = current_gen

        return gens

    def calc_shortest_path(self, p1, p2):
        layers = [[self.nodes[p1]]]
        self.nodes[p1].set_prev(None)
        current_layer = 0
        while p2 not in [x.index for l in layers for x in l]:
            new_layer = []
            for node in layers[current_layer]:
                for neighbor in node.children:
                    if neighbor.index not in [x.index for l in layers for x in l]:
                        neighbor.set_prev(node)
                        new_layer.append(neighbor)
            if len(new_layer) > 0:
                layers.append(new_layer)
            current_layer += 1
            if current_layer >= len(layers):
                return False
        return self.get_path_recursive(self.nodes[p2])

    def get_path_recursive(self, node, path=None):
        path = [] if path is None else path
        if node.prev is None:
            return [node.index]+path
        return self.get_path_recursive(node.prev, [node.index]+path)
