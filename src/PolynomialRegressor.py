from Matrix import Matrix
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


class PolynomialRegressor:
    def __init__(self, degree):
        self.degree = degree
        self.coefficients = [0 for _ in range(degree + 1)]

    def evaluate(self, x):
        return sum(self.coefficients[i] * x**i for i in range(len(self.coefficients)))

    def ingest_data(self, data):
        self.data = data

    def sum_squared_error(self):
        return sum((self.evaluate(x)-y)**2 for x, y in self.data)

    def solve_coefficients(self):
        x_data, y_data = [list(t) for t in zip(*self.data)]
        x_elems = [[t**i for i in range(0, self.degree+1)] for t in x_data]
        x_mat = Matrix(x_elems)
        y_mat = Matrix([y_data]).transpose()
        x_mat_t = x_mat.transpose()
        mul = x_mat_t @ x_mat
        inv = mul.inverse()
        result = inv @ (x_mat_t @ y_mat)
        self.coefficients = result[:, 0]

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
