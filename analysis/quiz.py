import sys
import math
sys.path.append("src")
try:
    from linear_regressor import LinearRegressor
    from matrix import Matrix
    from dataframe import DataFrame
except ImportError as e:
    print(e)

data = [(2, 27.0154), (3, 64.0912), (4, 159.817)]

df = DataFrame.from_array(data, ["x", "y"])
x_col = df.data_dict['x']
df = df.append_columns({
    'l': [math.log(x) for x in x_col],
    'm': [math.exp(x) for x in x_col],
    'n': [(x+1)**0.5 for x in x_col]
})
df = df.filter_columns(['l', 'm', 'n', 'y'])

regressor = LinearRegressor(df, 'y')
regressor.round_coefficients()
print("Coefficients Part A", regressor.coefficients)
