import sys
sys.path.append('src')
try:
    from polynomial_regressor import PolynomialRegressor
except ImportError as err:
    print(err)

data = [(1, 3.1), (2, 10.17)]

regressor = PolynomialRegressor(2)
regressor.ingest_data(data)
regressor.solve_coefficients()
prediction = regressor.evaluate(200)


cubic_regressor = PolynomialRegressor(3)
cubic_regressor.ingest_data(data)
cubic_regressor.solve_coefficients()
cubic_prediction = cubic_regressor.evaluate(200)

print("Quadratic Regression:", prediction)
print("Cubic Regression:", cubic_prediction)

# The Cubic Regression is going to be a more accurate answer than the
# Quadratic Regression because the cubic function is able to fit more
# lines than lower degree lines, and higher degree lines like quartic
# Can fit the line even better than the cubic
