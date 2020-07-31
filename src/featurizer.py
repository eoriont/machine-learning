class Featurizer:
    def __init__(self):
        self.columns = []

    def add_column(self, arr):
        self.columns.append(arr)

    def multiply_columns(self, col1, col2):
        return [x * y for x, y in zip(self.columns[col1], self.columns[col2])]

    def pop_column(self):
        return self.columns.pop()

    def insert_column(self, col, array):
        self.columns.insert(col, array)

    def get_matrix_elements(self):
        # Transpose self.columns
        return [arr for arr in zip(*self.columns)]
