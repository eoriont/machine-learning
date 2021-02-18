class TreeNode:
    # Constructor
    def __init__(self, data, children=None, parent=None):
        self.data = data
        self.children = [] if children == None else children
        self.parent = parent

    # Append a node to self.children
    def add_child(self, data):
        self.children.append(TreeNode(data, parent=self))

    # Return the node in the tree with data as it's self.data
    def get_node_by_value(self, data):
        if self.data == data:
            return self
        for child in self.children:
            child_search = child.get_node_by_value(data)
            if child_search is not None:
                return child_search
        return None

    def is_full(self):
        return len(self.children) == 2

    def next_append_child(self):
        if self.parent == None:
            if len(self.children) == 0:
                return None
            return self.children[0]
        elif self == self.parent.children[0]:
            return self.parent.children[1]
        elif self == self.parent.children[1]:
            return self.parent.next_append_child().children[0]

    def depth_first_search(self):
        return [self.data] + [data for child in self.children for data in child.depth_first_search()]
