import math
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
sys.path.append("src")
try:
    from deprecated_polynomial_regressor import PolynomialRegressor
except ImportError as e:
    print(e)


# Fit y to the function
def transform_y(y, _):
    return math.log((1/y)-1)

# Evaluate (inverse of transform_y)


def evaluate(betas, x):
    return 1/(1 + math.exp(betas[0] + betas[1]*x))


# This is the given data
data = [(0, 0.01), (0, 0.01), (0, 0.05), (10, 0.02), (10, 0.15), (50, 0.12), (50, 0.28), (73, 0.03), (80, 0.10), (115, 0.06), (150, 0.12), (170, 0.30), (175, 0.24),
        (198, 0.26), (212, 0.25), (232, 0.32), (240, 0.45), (381, 0.93), (390, 0.87), (402, 0.95), (450, 0.98), (450, 0.85), (450, 0.95), (460, 0.91), (500, 0.95)]

# Attach a 0th term
regressor_data = [(1, x, y) for x, y in data]

# Regress the data
regressor = PolynomialRegressor(1, transform_y)
regressor.ingest_data(regressor_data)
regressor.solve_coefficients()
betas = regressor.coefficients

# Problem A
print(f"Expected chance of winning against an average player if you have practiced for 300 hours:",
      round(evaluate(betas, 300), 3))

# Problem B
x_vals, y_vals = zip(*((i, evaluate(betas, i)) for i in range(750)))
plt.plot(x_vals, y_vals, zorder=1)
plt.title("Predicted y vals from 0 <= y <= 750")
plt.savefig("logistic_regression_space_empires")
