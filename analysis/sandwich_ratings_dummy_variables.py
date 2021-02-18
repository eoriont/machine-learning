import sys
sys.path.append("src/models")
sys.path.append("tests")
from matrix import Matrix
from featurizer import Featurizer
from otest import do_assert


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
f.add_column(ratings)
print(Matrix(f.get_matrix_elements()))

# Interaction features
f.insert_column(-1, f.multiply_columns(0, 1))
f.insert_column(-1, f.multiply_columns(0, 2))
f.insert_column(-1, f.multiply_columns(0, 3))
f.insert_column(-1, f.multiply_columns(1, 2))
f.insert_column(-1, f.multiply_columns(1, 3))
f.insert_column(-1, f.multiply_columns(2, 3))
X = f.get_matrix_elements()
print(Matrix(X))

do_assert("data input", sum(sum(x) for x in X), 313)

f.insert_column(0, [1 for _ in range(len(roast_beef))])
f.pop_column()
X = Matrix(f.get_matrix_elements())
y = Matrix([ratings]).transpose()
betas = ((X.transpose() @ X).inverse() @
         (X.transpose() @ y)).transpose().elements[0]

var_names = ["bias term", "beef", "peanut butter", "mayo", "jelly", "beef & peanut butter",
             "beef & mayo", "beef & jelly", "peanut butter & mayo", "peanut butter & jelly"]
for b, name in zip(betas, var_names):
    print(name, ":", b)


def predict(inputs, betas):
    return sum(x * y for x, y in zip(inputs, betas))


print("-----\nRATINGS\n------")
#              1  B  P  M  J  BP BM BJ PM PJ MJ
print("2 slices beef + mayo:",
      predict([1, 2, 0, 1, 0, 0, 2, 0, 0, 0, 0], betas))
print("2 slices beef + jelly:",
      predict([1, 2, 0, 0, 1, 0, 0, 2, 0, 0, 0], betas))
print("3 tbsp peanut butter + jelly:",
      predict([1, 0, 3, 0, 1, 0, 0, 0, 0, 3, 0], betas))
print("3 tbsp peanut butter + jelly + mayo:",
      predict([1, 0, 3, 1, 1, 0, 0, 0, 3, 3, 1], betas))
print("2 slices beef + 3 tbsp peanut butter + jelly + mayo:",
      predict([1, 2, 3, 1, 1, 6, 2, 2, 3, 3, 1], betas))
