import sys
sys.path.append("src")
try:
    from otest import do_assert, color_print
    from dataframe import DataFrame
    from linear_regressor import LinearRegressor
except ImportError as e:
    print(e)


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
linear_regressor = LinearRegressor(df, prediction_column='rating')
do_assert("linear regressor coefficients", _round(linear_regressor.coefficients),
          {
              'beef': 0.25,
              'pb': 0.4,
              'mayo': -1.25,
              'jelly': 1.5,
              'beef_pb': -0.21,
              'beef_mayo': 1.05,
              'beef_jelly': -0.85,
              'pb_mayo': -0.65,
              'pb_jelly': 0.65,
              'mayo_jelly': -3.25,
              'constant': 2.1875
})

do_assert("gather_all_inputs", linear_regressor.gather_all_inputs({
    'beef': 5,
    'pb': 5,
    'mayo': 1,
    'jelly': 1,
    'constant': 1
}),

    {
    'beef': 5,
    'pb': 5,
    'mayo': 1,
    'jelly': 1,
    'beef_pb': 25,
    'beef_mayo': 5,
    'beef_jelly': 5,
    'pb_mayo': 5,
    'pb_jelly': 5,
    'mayo_jelly': 1,
    'constant': 1
})


# ^ Note: this should include all interactions terms AND any constant term
# that also appear in the base dataframe

do_assert("prediction 1", round(linear_regressor.predict({
    'beef': 5,
    'pb': 5,
    'mayo': 1,
    'jelly': 1,
    'constant': 1
}), 5),
    -1.8125)

do_assert("prediction 2", round(linear_regressor.predict({
    'beef': 0,
    'pb': 3,
    'mayo': 0,
    'jelly': 1,
    'constant': 1
}), 5),
    6.8375)

do_assert("prediction 3", round(linear_regressor.predict({
    'beef': 1,
    'pb': 1,
    'mayo': 1,
    'jelly': 0,
    'constant': 1
}), 5),
    1.7775)

do_assert("prediction 4", round(linear_regressor.predict({
    'beef': 6,
    'pb': 0,
    'mayo': 1,
    'jelly': 0,
    'constant': 1
}), 5),
    8.7375)

print(linear_regressor.predict({
    'beef': 0,
    'pb': 0,
    'mayo': 0,
    'jelly': 0,
    'constant': 1
}))
