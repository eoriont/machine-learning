from weighted_node import WeightedNode

class WeightedGraph:
    def __init__(self, weights, vertices):
        self.weights = weights
        self.vertices = vertices
        self.nodes = [WeightedNode(value=v, index=i) for i, v in enumerate(vertices)]
        for (x, y), w in weights.items():
            self.nodes[x].set_neighbor(self.nodes[y], w)

    def calc_tdists(self, n1, n2):
        node = self.nodes[n1]
        s = [node]
        visited_nodes = []
        tdists = {x: float('inf') for x in range(len(self.vertices))}
        tdists[node.index] = 0
        paths = {x: [] for x in range(len(self.vertices))}
        while len(s) > 0:
            n = s.pop(0)
            paths[n.index].append(n.index)
            unvisited_neighbors = [x for x in n.neighbors if x not in visited_nodes]
            for nb in unvisited_neighbors:
                new_tdist = n.neighbors[nb] + tdists[n.index]
                tdists[nb.index] = min(new_tdist, tdists[nb.index])
                if new_tdist == tdists[nb.index] and nb not in visited_nodes:
                    paths[nb.index] = paths[n.index].copy()
            visited_nodes.append(n)
            s += [x for x in unvisited_neighbors if x not in s]
            s.sort(key=lambda x: tdists[x.index])
        return tdists, paths

    def calc_distance(self, n1, n2):
        return self.calc_tdists(n1, n2)[0][n2]

    def calc_shortest_path(self, n1, n2):
        return self.calc_tdists(n1, n2)[1][n2]
