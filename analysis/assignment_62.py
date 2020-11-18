import random
import sys
sys.path.append("src")
try:
    from dataframe import DataFrame
    from linear_regressor import LinearRegressor
except ImportError as e:
    print(e)

dataset = [(i/10, 3 + 0.005*i**2 + random.randrange(-5, 5))
           for i in range(100)]

training_indices = random.sample(range(len(dataset)), int(0.8*len(dataset)))
training_set = [x for i, x in enumerate(dataset) if i in training_indices]
testing_set = [x for i, x in enumerate(dataset) if i not in training_indices]

print("TRAINING RSS")
linear_df = DataFrame.from_array(training_set, ["x", 'y']).append_columns(
    {'const': [1 for _ in range(len(training_set))]})
linear_reg = LinearRegressor(linear_df, 'y')
print(linear_reg.rss(linear_df))
quadratic_df = linear_df.append_columns(
    {"x^2": [x**2 for x in linear_df.get_column('x')]})
quadratic_reg = LinearRegressor(quadratic_df, 'y')
print(quadratic_reg.rss(quadratic_df))
cubic_df = quadratic_df.append_columns(
    {"x^3": [x**3 for x in linear_df.get_column('x')]})
cubic_reg = LinearRegressor(cubic_df, 'y')
print(cubic_reg.rss(cubic_df))
quartic_df = cubic_df.append_columns(
    {"x^4": [x**4 for x in linear_df.get_column('x')]})
quartic_reg = LinearRegressor(quartic_df, 'y')
print(quartic_reg.rss(quartic_df))
quintic_df = quartic_df.append_columns(
    {"x^5": [x**5 for x in linear_df.get_column('x')]})
quintic_reg = LinearRegressor(quintic_df, 'y')
print(quintic_reg.rss(quintic_df))

print("TESTING RSS")
linear_df = DataFrame.from_array(testing_set, ["x", 'y']).append_columns(
    {'const': [1 for _ in range(len(training_set))]})
print(linear_reg.rss(linear_df))
quadratic_df = linear_df.append_columns(
    {"x^2": [x**2 for x in linear_df.get_column('x')]})
print(quadratic_reg.rss(quadratic_df))
cubic_df = quadratic_df.append_columns(
    {"x^3": [x**3 for x in linear_df.get_column('x')]})
print(cubic_reg.rss(cubic_df))
quartic_df = cubic_df.append_columns(
    {"x^4": [x**4 for x in linear_df.get_column('x')]})
print(quartic_reg.rss(quartic_df))
quintic_df = quartic_df.append_columns(
    {"x^5": [x**5 for x in linear_df.get_column('x')]})
print(quintic_reg.rss(quintic_df))
