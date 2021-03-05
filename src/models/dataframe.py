import random

class DataFrame:
    def __init__(self, data_dict, column_order=None):
        self.data_dict = data_dict
        self.columns = list(
            data_dict) if column_order is None else column_order

    def to_array(self):
        if self.columns[0] in self.data_dict:
            arrs = [self.data_dict[col] for col in self.columns]
            return [list(arr) for arr in zip(*arrs)]
        else:
            return []

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
        for x in self.columns:
            for y in self.columns:
                if x != y and (y, x) not in l:
                    l.append((x, y))
        return l

    @classmethod
    def from_array(cls, arr, cols):
        return DataFrame({c: list(a) for c, a in zip(cols, zip(*arr))}, cols)

    def filter_columns(self, cols):
        return DataFrame(self.data_dict, cols)

    def include_column(self, col):
        d, c = self.get_data_copies()
        return DataFrame(d, c+[col])

    def get_data_copies(self):
        return {k: v.copy() for k, v in self.data_dict.items()}, self.columns.copy()

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

    # This is the same as filter_columns
    def select_columns(self, cols):
        return DataFrame(self.data_dict, cols)

    def select_rows(self, rows):
        arr = self.to_array()
        return DataFrame.from_array((x for i, x in enumerate(arr) if i in rows), self.columns)

    def select_rows_where(self, func):
        arr = self.to_array()
        return DataFrame.from_array((x for x in arr if func(self.to_entry(x))), self.columns)

    def to_entry(self, arr):
        return {k: v for k, v in zip(self.columns, arr)}

    def add_entry(self, entry):
        d, c = self.get_data_copies()
        for col in c:
            d[col].append(entry[col])
        return DataFrame(d, c)

    def order_by(self, col, ascending=True):
        return DataFrame.from_array(sorted(
            self.to_array(),
            key=lambda x: x[self.columns.index(col)],
            reverse=not ascending
        ), self.columns)

    def get_column(self, col):
        if col in self.columns:
            if col in self.data_dict.keys():
                return self.data_dict[col].copy()
            else:
                return []
        else:
            raise Exception("Accessing column that doesn't exist!")

    def __len__(self):
        try:
            return len(next(iter(self.data_dict.values())))
        except StopIteration:
            return 0

    def remove_entry(self, index):
        d, c = self.get_data_copies()
        entry = {}
        for col in c:
            entry[col] = d[col][index]
            del d[col][index]
        return DataFrame(d, c), entry

    def to_json(self):
        return [self.to_entry(arr) for arr in self.to_array()]

    def remove_duplicates(self, col):
        indices = []
        checked = []
        for i, x in enumerate(self.data_dict[col]):
            if x not in checked:
                checked.append(x)
                indices.append(i)
        return self.select_rows(indices)

    def round_column(self, col, places):
        d, c = self.get_data_copies()
        for i, x in enumerate(d[col]):
            d[col][i] = round(x, places)
        return DataFrame(d, c)

    def random_row(self):
        return random.choice(self.to_array())

    @classmethod
    def from_csv(self, path_to_csv, header=True):
        with open(path_to_csv, "r") as file:
            s = file.read()
            data = [[]]
            for x in _reader(s):
                data[-1].append(x[1])
                if x[0] == "\n":
                    data.append([])
            return DataFrame.from_array(data[1:-1], data[0])

    def min(self, col):
        return min(self.data_dict[col])

    def max(self, col):
        return max(self.data_dict[col])

    def save_csv(self, path_to_csv):
        with open(path_to_csv, "w") as file:
            for i, col in enumerate(self.columns):
                file.write(str(col))
                if i+1 < len(self.columns):
                    file.write(",")
            file.write("\n")
            for row in self.to_array():
                for i, val in enumerate(row):
                    file.write(str(val))
                    if i+1 < len(row):
                        file.write(",")
                file.write("\n")


def _reader(s):
    in_str = False
    current_val = ""
    for x in s:
        if x == "\"":
            in_str = not in_str
        if not in_str and x in (",", "\n"):
            yield x, _parse_val(current_val)
            current_val = ""
        else:
            current_val += x

def _parse_val(v):
    v = v.strip()
    if len(v) == 0: return ""
    try: return int(v)
    except ValueError:
        try: return float(v)
        except ValueError:
            if v[0] == "\"": return v[1:-1]
            else: return str(v)
