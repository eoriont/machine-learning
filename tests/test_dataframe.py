import sys
import os
sys.path.append("src")
try:
    from otest import do_assert, color_print
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

data_dict = {
    'id': [1, 2, 3, 4],
    'color': ['blue', 'yellow', 'green', 'yellow']
}

df1 = DataFrame(data_dict, column_order=['id', 'color'])
df2 = df1.create_all_dummy_variables()
do_assert("create dummy variables", df2.columns,
          ['id', 'color_blue', 'color_yellow', 'color_green'])
do_assert("create dummy variables data", df2.to_array(),
          [[1, 1, 0, 0],
           [2, 0, 1, 0],
           [3, 0, 0, 1],
           [4, 0, 1, 0]])
df3 = df2.remove_columns(['id', 'color_yellow'])
do_assert("remove columns cols", df3.columns,
          ['color_blue', 'color_green'])
do_assert("remove columns data", df3.to_array(),
          [[1, 0],
           [0, 0],
           [0, 1],
           [0, 0]])
df4 = df3.append_columns({
    'name': ['Anna', 'Bill', 'Cayden', 'Daphnie'],
    'letter': ['a', 'b', 'c', 'd']
})
do_assert("append columns cols", df4.columns,
          ['color_blue', 'color_green', 'name', 'letter'])
do_assert("append columns data", df4.to_array(),
          [[1, 0, 'Anna', 'a'],
           [0, 0, 'Bill', 'b'],
           [0, 1, 'Cayden', 'c'],
           [0, 0, 'Daphnie', 'd']])
# !
color_print("Onto new tests... (Assignment 40-2)", "Cyan")
data_dict = {
    'beef': [0, 0, 0, 0, 5, 5, 5, 5, 0, 0, 0, 0, 5, 5, 5, 5],
    'pb': [0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5],
    'condiments': [[], ['mayo'], ['jelly'], ['mayo', 'jelly'],
                   [], ['mayo'], ['jelly'], ['mayo', 'jelly'],
                   [], ['mayo'], ['jelly'], ['mayo', 'jelly'],
                   [], ['mayo'], ['jelly'], ['mayo', 'jelly']],
    'rating': [1, 1, 4, 0, 4, 8, 1, 0, 5, 0, 9, 0, 0, 0, 0, 0]
}
df = DataFrame(data_dict, column_order=['beef', 'pb', 'condiments'])
do_assert("columns", df.columns, ['beef', 'pb', 'condiments'])
do_assert("matrix", df.to_array(),    [[0,  0,  []],
                                       [0,  0,  ['mayo']],
                                       [0,  0,  ['jelly']],
                                       [0,  0,  ['mayo', 'jelly']],
                                       [5,  0,  []],
                                       [5,  0,  ['mayo']],
                                       [5,  0,  ['jelly']],
                                       [5,  0,  ['mayo', 'jelly']],
                                       [0,  5,  []],
                                       [0,  5,  ['mayo']],
                                       [0,  5,  ['jelly']],
                                       [0,  5,  ['mayo', 'jelly']],
                                       [5,  5,  []],
                                       [5,  5,  ['mayo']],
                                       [5,  5,  ['jelly']],
                                       [5,  5,  ['mayo', 'jelly']]])

df = df.create_all_dummy_variables()
do_assert("dummy vars columns", df.columns, ['beef', 'pb', 'mayo', 'jelly'])
do_assert("dummy vars matrix", df.to_array(),   [[0,  0,  0,  0],
                                                 [0,  0,  1,  0],
                                                 [0,  0,  0,  1],
                                                 [0,  0,  1,  1],
                                                 [5,  0,  0,  0],
                                                 [5,  0,  1,  0],
                                                 [5,  0,  0,  1],
                                                 [5,  0,  1,  1],
                                                 [0,  5,  0,  0],
                                                 [0,  5,  1,  0],
                                                 [0,  5,  0,  1],
                                                 [0,  5,  1,  1],
                                                 [5,  5,  0,  0],
                                                 [5,  5,  1,  0],
                                                 [5,  5,  0,  1],
                                                 [5,  5,  1,  1]])

df = df.append_pairwise_interactions()
do_assert("pairwise interactions cols", df.columns, ['beef', 'pb', 'mayo', 'jelly',
                                                     'beef_pb', 'beef_mayo', 'beef_jelly',
                                                     'pb_mayo', 'pb_jelly',
                                                     'mayo_jelly'])
do_assert("pairwise interactions matrix", df.to_array(),
          [[0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
           [0,  0,  1,  0,  0,  0,  0,  0,  0,  0],
           [0,  0,  0,  1,  0,  0,  0,  0,  0,  0],
           [0,  0,  1,  1,  0,  0,  0,  0,  0,  1],
           [5,  0,  0,  0,  0,  0,  0,  0,  0,  0],
           [5,  0,  1,  0,  0,  5,  0,  0,  0,  0],
           [5,  0,  0,  1,  0,  0,  5,  0,  0,  0],
           [5,  0,  1,  1,  0,  5,  5,  0,  0,  1],
           [0,  5,  0,  0,  0,  0,  0,  0,  0,  0],
           [0,  5,  1,  0,  0,  0,  0,  5,  0,  0],
           [0,  5,  0,  1,  0,  0,  0,  0,  5,  0],
           [0,  5,  1,  1,  0,  0,  0,  5,  5,  1],
           [5,  5,  0,  0, 25,  0,  0,  0,  0,  0],
           [5,  5,  1,  0, 25,  5,  0,  5,  0,  0],
           [5,  5,  0,  1, 25,  0,  5,  0,  5,  0],
           [5,  5,  1,  1, 25,  5,  5,  5,  5,  1]])

df = df.append_columns({
    'constant': [1 for _ in range(len(data_dict['rating']))],
    'rating': data_dict['rating']
}, column_order=['constant', 'rating'])
do_assert("append_columns cols", df.columns, ['beef', 'pb', 'mayo', 'jelly',
                                              'beef_pb', 'beef_mayo', 'beef_jelly',
                                              'pb_mayo', 'pb_jelly',
                                              'mayo_jelly',
                                              'constant', 'rating'])
do_assert("append columns matrix", df.to_array(),   [[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1],
                                                     [0,  0,  1,  0,  0,  0,
                                                         0,  0,  0,  0,  1,  1],
                                                     [0,  0,  0,  1,  0,  0,
                                                         0,  0,  0,  0,  1,  4],
                                                     [0,  0,  1,  1,  0,  0,
                                                         0,  0,  0,  1,  1,  0],
                                                     [5,  0,  0,  0,  0,  0,
                                                         0,  0,  0,  0,  1,  4],
                                                     [5,  0,  1,  0,  0,  5,
                                                         0,  0,  0,  0,  1,  8],
                                                     [5,  0,  0,  1,  0,  0,
                                                         5,  0,  0,  0,  1,  1],
                                                     [5,  0,  1,  1,  0,  5,
                                                         5,  0,  0,  1,  1,  0],
                                                     [0,  5,  0,  0,  0,  0,
                                                         0,  0,  0,  0,  1,  5],
                                                     [0,  5,  1,  0,  0,  0,
                                                         0,  5,  0,  0,  1,  0],
                                                     [0,  5,  0,  1,  0,  0,
                                                         0,  0,  5,  0,  1,  9],
                                                     [0,  5,  1,  1,  0,  0,
                                                         0,  5,  5,  1,  1,  0],
                                                     [5,  5,  0,  0, 25,  0,
                                                         0,  0,  0,  0,  1,  0],
                                                     [5,  5,  1,  0, 25,  5,
                                                         0,  5,  0,  0,  1,  0],
                                                     [5,  5,  0,  1, 25,  0,
                                                         5,  0,  5,  0,  1,  0],
                                                     [5,  5,  1,  1, 25,  5,  5,  5,  5,  1,  1,  0]])

################################################################
#!==============================================================
# >============================================================

columns = ['firstname', 'lastname', 'age']
arr = [['Kevin', 'Fray', 5],
       ['Charles', 'Trapp', 17],
       ['Anna', 'Smith', 13],
       ['Sylvia', 'Mendez', 9]]
df = DataFrame.from_array(arr, columns)
do_assert("select columns by name", df.select_columns(['firstname', 'age']).to_array(),
          [['Kevin', 5],
           ['Charles', 17],
           ['Anna', 13],
           ['Sylvia', 9]])

do_assert("select row by index", df.select_rows([1, 3]).to_array(),
          [['Charles', 'Trapp', 17],
           ['Sylvia', 'Mendez', 9]])

do_assert("select row by condition", df.select_rows_where(
    lambda row: len(row['firstname']) >= len(row['lastname'])
    and row['age'] > 10
).to_array(),
    [['Charles', 'Trapp', 17]])


do_assert("order_rows_by_column", df.order_by('age', ascending=True).to_array(),
          [['Kevin', 'Fray', 5],
           ['Sylvia', 'Mendez', 9],
           ['Anna', 'Smith', 13],
           ['Charles', 'Trapp', 17]])

do_assert("order rows by column 2", df.order_by('firstname', ascending=False).to_array(),
          [['Sylvia', 'Mendez', 9],
           ['Kevin', 'Fray', 5],
           ['Charles', 'Trapp', 17],
           ['Anna', 'Smith', 13]])

# >=========================================================================================================
#!=========================================================================================================
# >=========================================================================================================


columns = ['Cookie Type', 'Portion Eggs',
           'Portion Butter', 'Portion Sugar', 'Portion Flour']
data = [['Shortbread',     0.14,       0.14,      0.28,     0.44],
        ['Shortbread',     0.10,       0.18,      0.28,     0.44],
        ['Shortbread',     0.12,       0.10,      0.33,     0.45],
        ['Shortbread',     0.10,       0.25,      0.25,     0.40],
        ['Sugar',     0.00,       0.10,      0.40,     0.50],
        ['Sugar',     0.00,       0.20,      0.40,     0.40],
        ['Sugar',     0.02,       0.08,      0.45,     0.45],
        ['Sugar',     0.10,       0.15,      0.35,     0.40],
        ['Sugar',     0.10,       0.08,      0.35,     0.47],
        ['Sugar',     0.00,       0.05,      0.30,     0.65],
        ['Fortune',     0.20,       0.00,      0.40,     0.40],
        ['Fortune',     0.25,       0.10,      0.30,     0.35],
        ['Fortune',     0.22,       0.15,      0.50,     0.13],
        ['Fortune',     0.15,       0.20,      0.35,     0.30],
        ['Fortune',     0.22,       0.00,      0.40,     0.38],
        ['Shortbread',     0.05,       0.12,      0.28,     0.55],
        ['Shortbread',     0.14,       0.27,      0.31,     0.28],
        ['Shortbread',     0.15,       0.23,      0.30,     0.32],
        ['Shortbread',     0.20,       0.10,      0.30,     0.40]]
df = DataFrame.from_array(data, columns)
json_data = df.to_json()
do_assert("JSON conversion", json_data, [{'Cookie Type': 'Shortbread', 'Portion Eggs': 0.14, 'Portion Butter': 0.14, 'Portion Sugar': 0.28, 'Portion Flour': 0.44},
                                         {'Cookie Type': 'Shortbread', 'Portion Eggs': 0.1,
                                             'Portion Butter': 0.18, 'Portion Sugar': 0.28, 'Portion Flour': 0.44},
                                         {'Cookie Type': 'Shortbread', 'Portion Eggs': 0.12,
                                             'Portion Butter': 0.1, 'Portion Sugar': 0.33, 'Portion Flour': 0.45},
                                         {'Cookie Type': 'Shortbread', 'Portion Eggs': 0.1,
                                             'Portion Butter': 0.25, 'Portion Sugar': 0.25, 'Portion Flour': 0.4},
                                         {'Cookie Type': 'Sugar', 'Portion Eggs': 0.0, 'Portion Butter': 0.1,
                                          'Portion Sugar': 0.4, 'Portion Flour': 0.5},
                                         {'Cookie Type': 'Sugar', 'Portion Eggs': 0.0, 'Portion Butter': 0.2,
                                          'Portion Sugar': 0.4, 'Portion Flour': 0.4},
                                         {'Cookie Type': 'Sugar', 'Portion Eggs': 0.02, 'Portion Butter': 0.08,
                                          'Portion Sugar': 0.45, 'Portion Flour': 0.45},
                                         {'Cookie Type': 'Sugar', 'Portion Eggs': 0.1, 'Portion Butter': 0.15,
                                          'Portion Sugar': 0.35, 'Portion Flour': 0.4},
                                         {'Cookie Type': 'Sugar', 'Portion Eggs': 0.1, 'Portion Butter': 0.08,
                                          'Portion Sugar': 0.35, 'Portion Flour': 0.47},
                                         {'Cookie Type': 'Sugar', 'Portion Eggs': 0.0, 'Portion Butter': 0.05,
                                          'Portion Sugar': 0.3, 'Portion Flour': 0.65},
                                         {'Cookie Type': 'Fortune', 'Portion Eggs': 0.2, 'Portion Butter': 0.0,
                                          'Portion Sugar': 0.4, 'Portion Flour': 0.4},
                                         {'Cookie Type': 'Fortune', 'Portion Eggs': 0.25, 'Portion Butter': 0.1,
                                          'Portion Sugar': 0.3, 'Portion Flour': 0.35},
                                         {'Cookie Type': 'Fortune', 'Portion Eggs': 0.22, 'Portion Butter': 0.15,
                                          'Portion Sugar': 0.5, 'Portion Flour': 0.13},
                                         {'Cookie Type': 'Fortune', 'Portion Eggs': 0.15, 'Portion Butter': 0.2,
                                          'Portion Sugar': 0.35, 'Portion Flour': 0.3},
                                         {'Cookie Type': 'Fortune', 'Portion Eggs': 0.22, 'Portion Butter': 0.0,
                                          'Portion Sugar': 0.4, 'Portion Flour': 0.38},
                                         {'Cookie Type': 'Shortbread', 'Portion Eggs': 0.05,
                                          'Portion Butter': 0.12, 'Portion Sugar': 0.28, 'Portion Flour': 0.55},
                                         {'Cookie Type': 'Shortbread', 'Portion Eggs': 0.14,
                                          'Portion Butter': 0.27, 'Portion Sugar': 0.31, 'Portion Flour': 0.28},
                                         {'Cookie Type': 'Shortbread', 'Portion Eggs': 0.15,
                                          'Portion Butter': 0.23, 'Portion Sugar': 0.3, 'Portion Flour': 0.32},
                                         {'Cookie Type': 'Shortbread', 'Portion Eggs': 0.2, 'Portion Butter': 0.1, 'Portion Sugar': 0.3, 'Portion Flour': 0.4}]
          )

# >===================================================================================
# >===================================================================================
# >===================================================================================

path = os.path.join(os.getcwd(), 'datasets', 'airtravel.csv')
df = DataFrame.from_csv(path, header=True)

do_assert("to_csv", df.to_array(), [["Month", "1958", "1959", "1960"],
                                    ["JAN",  340,  360,  417],
                                    ["FEB",  318,  342,  391],
                                    ["MAR",  362,  406,  419],
                                    ["APR",  348,  396,  461],
                                    ["MAY",  363,  420,  472],
                                    ["JUN",  435,  472,  535],
                                    ["JUL",  491,  548,  622],
                                    ["AUG",  505,  559,  606],
                                    ["SEP",  404,  463,  508],
                                    ["OCT",  359,  407,  461],
                                    ["NOV",  310,  362,  390],
                                    ["DEC",  337,  405,  432]])
