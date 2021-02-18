from tree_node import TreeNode

class Tree:
    # Constructor
    def __init__(self, data, root=None):
        self.root = TreeNode(data) if root is None else root

    def __str__(self):
        return str(self.breadth_first_search())

    def append(self, data):
        current_node = self.root
        while current_node.is_full():
            current_node = current_node.next_append_child()

        current_node.add_child(data)

    def insert(self, new_tree, data):
        self.root.get_node_by_value(data).children.append(new_tree.root)

    def breadth_first_search(self):
        queue = [self.root.data]
        result = []
        while len(queue) > 0:
            next_val = queue.pop()
            next_node = self.root.get_node_by_value(next_val)
            queue[:0] = [node.data for node in next_node.children]
            result.append(next_val)
        return result

    def depth_first_search(self):
        return self.root.depth_first_search()

    @staticmethod
    def build_from_edges(edges):
        current_node = TreeNode(edges[0][0])
        for edge in edges:
            root = current_node.get_node_by_value(edge[0])
            if root is not None:
                root.add_child(edge[1])
            else:
                if edge[1] == current_node.data:
                    current_node = TreeNode(edge[0], children=[current_node])
                else:
                    edges.append(edge)
        return Tree(0, current_node)


print(Tree.build_from_edges([('b', 'a'), ('c', 'd'), ('e', 'b'),
                             ('f', 'e'), ('g', 'c'), ('h', 'f'), ('f', 'g'), ('h', 'b')]).root.data)
