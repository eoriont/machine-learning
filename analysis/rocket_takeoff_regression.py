import sys
sys.path.append('src/models')
from deprecated_polynomial_regressor import PolynomialRegressor

data = [(1, 3.1), (2, 10.17), (3, 20.93), (4, 38.71), (5, 60.91),
        (6, 98.87), (7, 113.92), (8, 146.95), (9, 190.09), (10, 232.65)]

regressor = PolynomialRegressor(2)
regressor.ingest_data(data)
regressor.solve_coefficients()
prediction = regressor.evaluate(200)


cubic_regressor = PolynomialRegressor(3)
cubic_regressor.ingest_data(data)
cubic_regressor.solve_coefficients()
cubic_prediction = cubic_regressor.evaluate(200)

print("Quadratic Coefficients:", regressor.coefficients)
print("Quadratic Regression:", prediction)
print("Cubic Coefficients:", cubic_regressor.coefficients)
print("Cubic Regression:", cubic_prediction)

# For this situation, the quadratic model is actually better than the
# cubic model, because the cubic model becomes negative after enough
# time. In general though, the cubic should fit the lines better
# because of the increased amount of coefficients which allow for more
# detail.
