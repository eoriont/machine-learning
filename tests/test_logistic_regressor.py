import sys
sys.path.append("src/models")
from otest import do_assert
from dataframe import DataFrame
from logistic_regressor import LogisticRegressor


def _round(x):
    return {key: round(val, 5) for key, val in x.items()}


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
df = df.create_all_dummy_variables()
df = df.append_pairwise_interactions()
df = df.append_columns({
    'constant': [1 for _ in range(len(data_dict['rating']))],
    'rating': data_dict['rating']
})
df = df.apply('rating', lambda x: 0.1 if x == 0 else x)
regressor = LogisticRegressor(df, prediction_column='rating', max_value=10)
do_assert("coefficients", _round(regressor.coefficients),
          _round({
              'beef': -0.03900793,
              'pb': -0.02047944,
              'mayo': 1.74825378,
              'jelly': -0.39777219,
              'beef_pb': 0.14970983,
              'beef_mayo': -0.74854916,
              'beef_jelly': 0.46821312,
              'pb_mayo': 0.32958369,
              'pb_jelly': -0.5288267,
              'mayo_jelly': 2.64413352,
              'constant': 1.01248436
          }))

do_assert("prediction 1", round(regressor.predict({
    'beef': 5,
    'pb': 5,
    'mayo': 1,
    'jelly': 1,
    'constant': 1
}), 5),
    round(0.023417479576338825, 5))

do_assert("prediction 2", round(regressor.predict({
    'beef': 0,
    'pb': 3,
    'mayo': 0,
    'jelly': 1,
    'constant': 1
}), 5),
    round(7.375370259327203, 5))

do_assert("prediction 3", round(regressor.predict({
    'beef': 1,
    'pb': 1,
    'mayo': 1,
    'jelly': 0,
    'constant': 1
}), 5),
    round(0.8076522077650409, 5))

do_assert("prediction 4", round(regressor.predict({
    'beef': 6,
    'pb': 0,
    'mayo': 1,
    'jelly': 0,
    'constant': 1
}), 5),
    round(8.770303903540402, 5))
