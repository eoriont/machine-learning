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

    @classmethod
    def from_array(cls, arr, cols):
        return DataFrame.from_mat(arr, cols)

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

    def trim_column(self, column, max_num):
        d, c = self.get_data_copies()
        small_num = 0.2

        def t(x):
            if x == 0:
                return small_num
            elif x == max_num:
                return max_num - small_num
            else:
                return x
        d[column] = [t(x) for x in d[column]]
        return DataFrame(d, c)

    def select_columns(self, cols):
        return self.filter_columns(cols)

    def select_rows(self, rows):
        d, c = self.get_data_copies()
        d = {k: [x for i, x in enumerate(v) if i in rows]
             for k, v in d.items()}
        return DataFrame(d, c)

    def get_rows(self, func=lambda _: True):
        return [x for x in zip(*self.data_dict.values()) if func({k: v for k, v in zip(self.columns, x)})]

    def select_rows_where(self, func):
        d, c = self.get_data_copies()
        cols = list(zip(*self.get_rows(func)))
        d = {k: v for k, v in zip(c, cols)}
        return DataFrame(d, c)

    def add_entry(self, entry):
        d, c = self.get_data_copies()
        for col in c:
            d[col].appen(entry[col])
        return DataFrame(d, c)

    def order_by(self, col, ascending):
        d, c = self.get_data_copies()
        rows = self.get_rows()
        rows.sort(key=lambda x: x[c.index(col)], reverse=not ascending)
        cols = list(zip(*rows))
        d = {k: v for k, v in zip(c, cols)}
        return DataFrame(d, c)

    def get_column(self, col):
        return self.data_dict[col].copy()
