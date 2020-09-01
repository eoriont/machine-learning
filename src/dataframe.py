class DataFrame:
    def __init__(self, data_dict, column_order=None):
        self.data_dict = data_dict
        column_order = column_order if column_order is not None else list(
            data_dict.keys())
        self.columns = column_order

    def to_array(self):
        arrs = [self.data_dict[col] for col in self.columns]
        return [list(arr) for arr in zip(*arrs)]

    def filter_columns(self, cols):
        return DataFrame(self.data_dict, cols)
