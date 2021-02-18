class DirectedNode:
    def __init__(self, index, value):
        self.index = index
        self.value = value
        self.children = []
        self.parents = []
        self.prev = None

    def add_parent(self, p):
        self.parents.append(p)

    def add_child(self, c):
        self.children.append(c)

    def get_neighbors(self):
        return self.children + self.parents

    def set_prev(self, prev):
        self.prev = prev
