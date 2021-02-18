class Node:
    def __init__(self, index, value=0):
        self.index = index
        self.value = value
        self.neighbors = []
        self.parent = None

    def set_value(self, value):
        self.value = value

    def set_parent(self, parent):
        self.parent = parent

    def set_neighbor(self, neighbor):
        self.neighbors.append(neighbor)
        neighbor.neighbors.append(self)

    def depth_first_search(self, already_visited=None):
        already_visited.append(self.index)
        neighbor_indices = (neighbor.depth_first_search(already_visited)
                            for neighbor in self.neighbors if neighbor.index not in already_visited)
        return [self.index] + sum(neighbor_indices, [])
