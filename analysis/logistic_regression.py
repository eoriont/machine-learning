import sys
import math
sys.path.append("src")
try:
    from matrix import Matrix
    from deprecated_polynomial_regressor import PolynomialRegressor
except ImportError as e:
    print(e)


def main(data):
    data = [(1, *a) for a in data]
    print(data)
    regressor = PolynomialRegressor(1, y_function)
    regressor.ingest_data(data)
    regressor.solve_coefficients()
    betas = regressor.coefficients
    print("Betas:", betas)
    print("Probability of a 500hr player winning:",
          y_function_inverse(500, betas))
    print("The average player practices:", solve_for_x(.5, betas))


def solve_for_x(x, betas):
    return (y_function(x, 0) - betas[0])/betas[1]


def y_function(y, _):
    return math.log((1-y)/y)


def y_function_inverse(x, betas):
    return 1/(1+math.exp(betas[0] + x*betas[1]))


if __name__ == "__main__":
    data = [(10, 0.05), (100, .35), (1000, .95)]
    main(data)
