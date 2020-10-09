class DataFrame:
    def __init__(self, data_dict, column_order=None):
        self.data_dict = data_dict
        self.columns = list(
            data_dict) if column_order is None else column_order

    def to_array(self):
        arrs = [self.data_dict[col] for col in self.columns]
        return [list(arr) for arr in zip(*arrs)]

    def append_pairwise_interactions(self):
        columns_to_multiply = self.cartesian_product()
        d, c = self.get_data_copies()
        for x in columns_to_multiply:
            new_col = '_'.join(x)
            c.append(new_col)
            d[new_col] = [x*y for x, y in zip(d[x[0]], d[x[1]])]
        return DataFrame(d, c)

    def cartesian_product(self):
        l = []
        [l.append((x, y)) for x in self.columns for y in self.columns if x != y and (
            y, x) not in l]
        return l

    @staticmethod
    def from_mat(mat, cols):
        d = {col: list(arr) for col, arr in zip(cols, zip(*mat))}
        return DataFrame(d, cols)

    def filter_columns(self, cols):
        return DataFrame(self.data_dict, cols)

    def get_data_copies(self):
        return self.data_dict.copy(), self.columns.copy()

    def create_all_dummy_variables(self):
        categorical_cols = [col for col in self.data_dict if any(
            isinstance(i, str) for i in self.data_dict[col])]
        list_categorical_cols = [col for col in self.data_dict if any(
            isinstance(i, list) for i in self.data_dict[col])]
        df = self
        for col in categorical_cols:
            df = df.create_string_dummy_variables(col)
        for col in list_categorical_cols:
            df = df.create_list_dummy_variables(col)
        return df

    def create_string_dummy_variables(self, col):
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

    def create_list_dummy_variables(self, col):
        d, c = self.get_data_copies()
        col_data = []
        for arr in d[col]:
            for x in arr:
                if x not in col_data:
                    col_data.append(x)
        new_data = [[(1 if item in arr else 0) for arr in d[col]]
                    for item in col_data]
        new_cols_data = dict(zip(col_data, new_data))
        del d[col]
        d.update(new_cols_data)
        c.remove(col)
        c += col_data
        return DataFrame(d, c)

    def remove_columns(self, cols):
        d, c = self.get_data_copies()
        for col in cols:
            c.remove(col)
            del d[col]
        return DataFrame(d, c)

    def append_columns(self, data_dict, column_order=None):
        d, c = self.get_data_copies()
        d.update(data_dict)
        c += list(data_dict) if column_order is None else column_order
        return DataFrame(d, c)

    def apply(self, column, func):
        d, c = self.get_data_copies()
        d[column] = [func(x) for x in d[column]]
        return DataFrame(d, c)
