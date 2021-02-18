from matrix import Matrix
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


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

    def plot(self):
        x_data, y_data = zip(*self.data)
        y = [self.evaluate(x) for x in x_data]
        plt.scatter(x_data, y_data, zorder=2, color='black')
        x_max, y_max = 20, 20
        plt.xlim(-0.5, x_max-0.5)
        plt.ylim(-0.5, y_max-0.5)
        plt.grid(which='minor')
        plt.title("y = " + ' + '.join(f"{round(coef, 5)}x^{i}" for i,
                                      coef in enumerate(self.coefficients)))
        plt.plot(x_data, y, zorder=1)
        plt.show()
