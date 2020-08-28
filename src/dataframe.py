import sys
sys.path.append("tests")
try:
    from otest import do_assert
except ImportError as e:
    print(e)


class DataFrame:
    def __init__(self, data_dict, column_order):
        self.data_dict = data_dict
        self.columns = column_order

    def to_array(self):
        arrs = [self.data_dict[col] for col in self.columns]
        return [list(arr) for arr in zip(*arrs)]

    def filter_columns(self, cols):
        return DataFrame(self.data_dict, cols)


if __name__ == "__main__":
    data_dict = {
        'Pete': [1, 0, 1, 0],
        'John': [2, 1, 0, 2],
        'Sara': [3, 1, 4, 0]
    }

    df1 = DataFrame(data_dict, column_order=['Pete', 'John', 'Sara'])
    do_assert("Data dict is equal", df1.data_dict, data_dict)

    do_assert("expected array #1", df1.to_array(), [[1, 2, 3],
                                                    [0, 1, 1],
                                                    [1, 0, 4],
                                                    [0, 2, 0]])

    do_assert("Columns #1", df1.columns, ['Pete', 'John', 'Sara'])

    df2 = df1.filter_columns(['Sara', 'Pete'])
    do_assert("expected array #2", df2.to_array(), [[3, 1],
                                                    [1, 0],
                                                    [4, 1],
                                                    [0, 0]])

    do_assert("columns #2", df2.columns, ['Sara', 'Pete'])
