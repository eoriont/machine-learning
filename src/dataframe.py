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
        columns_to_multiply = self.cartesian_product(
            [self.columns, self.columns])
        d = self.data_dict.copy()
        c = self.columns.copy()
        for x in columns_to_multiply:
            new_col = '_'.join(x)
            c.append(new_col)
            d[new_col] = [x*y for x,
                          y in zip(self.data_dict[x[0]], self.data_dict[x[1]])]
        return DataFrame(d, c)

    def cartesian_product(self, arrays, points=None):
        points = [[]] if points is None else points
        new_points = [set(p+[consider]) for consider in arrays[0]
                      for p in points if consider not in p]
        filtered_points = []
        [filtered_points.append(x)
         for x in new_points if x not in filtered_points]
        filtered_points = [list(x) for x in filtered_points]

        return self.cartesian_product(arrays[1:], filtered_points) if len(arrays) > 1 else filtered_points

    def filter_columns(self, cols):
        return DataFrame(self.data_dict, cols)
