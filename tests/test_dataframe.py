import sys
sys.path.append("src")
try:
    from otest import do_assert
    from dataframe import DataFrame
except ImportError as e:
    print(e)

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
# =========================
# 38-2


data_dict = {
    'Pete': [1, 0, 1, 0],
    'John': [2, 1, 0, 2],
    'Sarah': [3, 1, 4, 0]
}

df1 = DataFrame(data_dict, column_order=['Pete', 'John', 'Sarah'])
df2 = df1.append_pairwise_interactions()
do_assert("pairwise_interactions columns", df2.columns, [
          'Pete', 'John', 'Sarah', 'Pete_John', 'Pete_Sarah', 'John_Sarah'])

do_assert("pairwise_interactions array", df2.to_array(), [[1, 2, 3, 2, 3, 6],
                                                          [0, 1, 1, 0, 0, 1],
                                                          [1, 0, 4, 0, 4, 0],
                                                          [0, 2, 0, 0, 0, 0]])
