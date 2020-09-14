import math
from matrix import Matrix


class PolynomialRegressor:
    def __init__(self, degree, y_function=None):
        self.degree = degree
        self.y_function = (lambda x: x) if y_function is None else y_function

    def evaluate(self, x):
        return sum(self.coefficients[i] * x**i for i in range(self.degree))

    def ingest_data(self, data):
        self.data = data

    def sum_squared_error(self):
        return sum((self.evaluate(x)-y)**2 for x, y in self.data)

    def solve_coefficients(self):
        x_data = [list(t) for t in zip(*self.data)]
        y_data = [x_data.pop()]
        X, y = Matrix(x_data).transpose(), Matrix(
            y_data).transpose().element_operation(self.y_function)
        result = (X.transpose() @ X).inverse() @ (X.transpose() @ y)

        self.coefficients = result[:, 0]
        if type(self.coefficients) != list:
            self.coefficients = [self.coefficients]
