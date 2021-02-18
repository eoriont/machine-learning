import sys
import math
sys.path.append("src/models")
from linear_regressor import LinearRegressor
from dataframe import DataFrame

#! Part A
data = [(0.0, 4.0),
        (0.2, 8.9),
        (0.4, 17.2),
        (0.6, 28.3),
        (0.8, 41.6),
        (1.0, 56.5),
        (1.2, 72.4),
        (1.4, 88.7),
        (1.6, 104.8),
        (1.8, 120.1),
        (2.0, 134.0),
        (2.2, 145.9),
        (2.4, 155.2),
        (2.6, 161.3),
        (2.8, 163.6),
        (3.0, 161.5),
        (3.2, 154.4),
        (3.4, 141.7),
        (3.6, 122.8),
        (3.8, 97.1),
        (4.0, 64.0),
        (4.2, 22.9),
        (4.4, -26.8),
        (4.6, -85.7),
        (4.8, -154.4)]

df = DataFrame.from_array(data, ["b", "y"])
x_col = df.data_dict['b']
df = df.append_columns({
    'c': [x**2 for x in x_col],
    'd': [x**3 for x in x_col]
})
df = df.append_columns({'a': [1 for _ in range(len(data))]})
df = df.filter_columns(['a', 'b', 'c', 'd', 'y'])

regressor = LinearRegressor(df, 'y')
regressor.round_coefficients()
print("Coefficients Part A", regressor.coefficients)


#! Part B
data = [(0.0, 7.0),
        (0.2, 5.6),
        (0.4, 3.56),
        (0.6, 1.23),
        (0.8, -1.03),
        (1.0, -2.89),
        (1.2, -4.06),
        (1.4, -4.39),
        (1.6, -3.88),
        (1.8, -2.64),
        (2.0, -0.92),
        (2.2, 0.95),
        (2.4, 2.63),
        (2.6, 3.79),
        (2.8, 4.22),
        (3.0, 3.8),
        (3.2, 2.56),
        (3.4, 0.68),
        (3.6, -1.58),
        (3.8, -3.84),
        (4.0, -5.76),
        (4.2, -7.01),
        (4.4, -7.38),
        (4.6, -6.76),
        (4.8, -5.22)]


df = DataFrame.from_array(data, ["x", "y"])
x_col = df.data_dict['x']
df = df.append_columns({
    'a': [math.sin(x) for x in x_col],
    'b': [math.cos(x) for x in x_col],
    'c': [math.sin(2*x) for x in x_col],
    'd': [math.cos(2*x) for x in x_col]
})
df = df.remove_columns(['x'])
df = df.filter_columns(['a', 'b', 'c', 'd', 'y'])

regressor = LinearRegressor(df, 'y')
print("Coefficients Part B", regressor.coefficients)
