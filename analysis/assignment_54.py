import math
import sys
sys.path.append('src')
try:
    from dataframe import DataFrame
    from logistic_regressor import LogisticRegressor
except ImportError as e:
    print(e)

data = [[1, 95, 33, 1, 0],
        [0, 95, 34, 0, 0],
        [1, 92, 35, 0, 1],
        [0, 85, 30, 0, 1],
        [1, 80, 36, 1, 0],
        [0, 85, 29, 1, 0],
        [1, 95, 36, 1, 0],
        [0, 87, 31, 1, 0],
        [1, 99, 36, 0, 0],
        [0, 95, 32, 0, 0]]

df = DataFrame.from_array(
    data, ['accepted', 'percentile', 'score', 'academicec', 'sportsec'])
df = df.append_columns({'const': [1 for _ in data]})
print(df.data_dict)

regressor = LogisticRegressor(df, 'accepted')
regressor.round_coefficients()
print(regressor.coefficients)

# # Martha
# print(regressor.predict(
#     {'percentile': 95, 'score': 33, 'academicec': 1, 'sportsec': 0, 'const': 1}))

# # Mary
# print(regressor.predict(
#     {'percentile': 95, 'score': 36, 'academicec': 1, 'sportsec': 0, 'const': 1}))

# # Denis
# print(regressor.predict(
#     {'percentile': 85, 'score': 30, 'academicec': 0, 'sportsec': 1, 'const': 1}))

# # Adam
# print(regressor.predict(
#     {'percentile': 99, 'score': 36, 'academicec': 0, 'sportsec': 0, 'const': 1}))

# print(regressor.predict({'percentile': 0, 'score': 0,
#                          'academicec': 0, 'sportsec': 0, 'const': 1}))

print(regressor.predict({'percentile': 95, 'score': 34,
                         'academicec': 0, 'sportsec': 0, 'const': 1}))
