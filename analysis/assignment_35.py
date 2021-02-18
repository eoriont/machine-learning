import sys
import math
sys.path.append("src/models")
from featurizer import Featurizer
from deprecated_polynomial_regressor import PolynomialRegressor


roast_beef = [0, 0, 0, 0, 5, 5, 5, 5, 0, 0, 0, 0, 5, 5, 5, 5]
peanut_butter = [0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5]
mayo = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
jelly = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
ratings = [1, 1, 4, 0, 4, 8, 1, 0, 5, 0, 9, 0, 0, 0, 0, 0]
f = Featurizer()
f.add_column(roast_beef)
f.add_column(peanut_butter)
f.add_column(mayo)
f.add_column(jelly)

# Interaction features
f.insert_column("end", f.multiply_columns(0, 1))
f.insert_column("end", f.multiply_columns(0, 2))
f.insert_column("end", f.multiply_columns(0, 3))
f.insert_column("end", f.multiply_columns(1, 2))
f.insert_column("end", f.multiply_columns(1, 3))
f.insert_column("end", f.multiply_columns(2, 3))
f.insert_column(0, [1 for _ in range(len(roast_beef))])
f.add_column(ratings)
f.remove_zeros(len(f.columns)-1)


# Function to transform y
def logistic_transform(y, _):
    return math.log((10/y)-1)


# Do regression with logistic transform
regressor = PolynomialRegressor(0, logistic_transform)
regressor.ingest_data(f.get_matrix_elements())
regressor.solve_coefficients()
betas = regressor.coefficients


# Just fit the inputs and betas to y (inverse of logistic transform)
def predict(inputs):
    return 10/(1+math.exp(sum(b * i for b, i in zip(betas, inputs))))


print("-----\nRATINGS\n------")
#              1  B  P  M  J  BP BM BJ PM PJ MJ
print("No ingredients:",
      predict([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
print("Mayo only",
      predict([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]))
print("Mayo and Jelly",
      predict([1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1]))
print("5 slices beef + mayo",
      predict([1, 5, 0, 1, 0, 0, 5, 0, 0, 0, 0]))
print("5 slices beef + 5 tbsp pb + mayo + jelly",
      predict([1, 5, 5, 1, 1, 25, 5, 5, 5, 5, 1]))
