import sys
sys.path.append("src/models")
from matrix import Matrix

#   1 * beta_0 + 0 * beta_1 + 0 * beta_2 = 1
#   1 * beta_0 + 1 * beta_1 + 0 * beta_2 = 2
#   1 * beta_0 + 2 * beta_1 + 0 * beta_2 = 4
#   1 * beta_0 + 4 * beta_1 + 0 * beta_2 = 8
#   1 * beta_0 + 6 * beta_1 + 0 * beta_2 = 9
#   1 * beta_0 + 0 * beta_1 + 2 * beta_2 = 2
#   1 * beta_0 + 0 * beta_1 + 4 * beta_2 = 5
#   1 * beta_0 + 0 * beta_1 + 6 * beta_2 = 7
#   1 * beta_0 + 0 * beta_1 + 8 * beta_2 = 6

#   [[1, 0, 0],                   [[1],
#    [1, 1, 0],                    [2],
#    [1, 2, 0],                    [4],
#    [1, 4, 0],    [[beta_0],      [8],
#    [1, 6, 0],     [beta_1],  =   [9],
#    [1, 0, 2],     [beta_2]]      [2],
#    [1, 0, 4],                    [5],
#    [1, 0, 6],                    [7],
#    [1, 0, 8]]                    [6]]

#   rating = beta_0 + beta_1 * (slices of beef) + beta_2 * (tbsp peanut butter)

X = Matrix([[1, 0, 0], [1, 1, 0], [1, 2, 0], [1, 4, 0], [
           1, 6, 0], [1, 0, 2], [1, 0, 4], [1, 0, 6], [1, 0, 8]])
y = Matrix([[1], [2], [4], [8], [9], [2], [5], [7], [6]])
betas = (X.transpose() @ X).inverse() @ (X.transpose() @ y)
betas = betas.transpose().elements[0]
print("Betas 0, 1, and 2 respectively:", betas)


def dot(arr1, arr2):
    return sum(x * y for x, y in zip(arr1, arr2))


print("Predicted Rating of 5 slices of beef and no peanut butter:",
      dot([1, 5, 0], betas))

print("Predicted Rating of 5 slices of roast beef and 5 scoops of peanut butter:",
      dot([1, 5, 5], betas))

# The company should DEFINITELY NOT trust the prediction, because the input data has no ties to eachother,
# meaning there isn't a singular piece of data where both roast beef and peanut butter are on the same sandwich.
# This means the prediction thinks both roast beef and peanut butter make the sandwich better, but in reality,
# having both on the same sandwich and eating it will send you directly to the hospital.
