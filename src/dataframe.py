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
        d, c = self.get_data_copies()
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

    def get_data_copies(self):
        return self.data_dict.copy(), self.columns.copy()

    def create_all_dummy_variables(self):
        categorical_cols = [col for col in self.data_dict if any(
            isinstance(i, str) for i in self.data_dict[col])]
        df = self
        for col in categorical_cols:
            df = df.create_dummy_variables(col)
        return df

    def create_dummy_variables(self, col):
        d, c = self.get_data_copies()
        col_data = []
        [col_data.append(i) for i in d[col] if i not in col_data]
        new_cols = [f"{col}_{name}" for name in col_data]
        new_data = [[(1 if i == item else 0)
                     for i in d[col]] for item in col_data]
        new_cols_data = dict(zip(new_cols, new_data))
        del d[col]
        d.update(new_cols_data)
        c.remove(col)
        c += new_cols
        return DataFrame(d, c)

    def remove_columns(self, cols):
        d, c = self.get_data_copies()
        for col in cols:
            c.remove(col)
            del d[col]
        return DataFrame(d, c)

    def append_columns(self, data_dict):
        d, c = self.get_data_copies()
        d.update(data_dict)
        c += list(data_dict)
        return DataFrame(d, c)


if __name__ == "__main__":
    data_dict = {
        'id': [1, 2, 3, 4],
        'color': ['blue', 'yellow', 'green', 'yellow']
    }

    df1 = DataFrame(data_dict, column_order=['id', 'color'])
    df2 = df1.create_all_dummy_variables()

    df3 = df2.remove_columns(['id', 'color_yellow'])
    print(df3.columns)
    print(df3.to_array())
    df4 = df3.append_columns({
        'name': ['Anna', 'Bill', 'Cayden', 'Daphnie'],
        'letter': ['a', 'b', 'c', 'd']
    })
    print(df4.columns)
    print(df4.to_array())
