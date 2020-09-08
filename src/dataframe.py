class DataFrame:
    def __init__(self, data_dict, column_order=None):
        self.data_dict = data_dict
        column_order = column_order if column_order is not None else list(
            data_dict.keys())
        self.columns = column_order

    def to_array(self):
        arrs = [self.data_dict[col] for col in self.columns]
        return [list(arr) for arr in zip(*arrs)]

    def append_pairwise_interactions(self):
        columns_to_multiply = self.cartesian_product()
        d = self.data_dict.copy()
        c = self.columns.copy()
        for x in columns_to_multiply:
            new_col = '_'.join(x)
            c.append(new_col)
            d[new_col] = [x*y for x,
                          y in zip(self.data_dict[x[0]], self.data_dict[x[1]])]
        return DataFrame(d, c)

    def cartesian_product(self):
        l = []
        [l.append((x, y)) for x in self.columns for y in self.columns if x != y and (
            y, x) not in l]
        return l

    def filter_columns(self, cols):
        return DataFrame(self.data_dict, cols)
