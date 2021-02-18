class WeightedNode:
    def __init__(self, value, index):
        self.value = value
        self.index = index
        # Neighbors is in format <index>:<weight>
        self.neighbors = {}

    def set_neighbor(self, neighbor, weight):
        self.neighbors[neighbor] = weight
        neighbor.neighbors[self] = weight
