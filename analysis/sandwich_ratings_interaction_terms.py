import sys
sys.path.append("src")
try:
    from matrix import Matrix
except ImportError as e:
    print(e)

#   1 * beta_0 + 0 * beta_1 + 0 * beta_2 = 1
#   1 * beta_0 + 1 * beta_1 + 0 * beta_2 = 2
#   1 * beta_0 + 2 * beta_1 + 0 * beta_2 = 4
#   1 * beta_0 + 4 * beta_1 + 0 * beta_2 = 8
#   1 * beta_0 + 6 * beta_1 + 0 * beta_2 = 9
#   1 * beta_0 + 0 * beta_1 + 2 * beta_2 = 2
#   1 * beta_0 + 0 * beta_1 + 4 * beta_2 = 5
#   1 * beta_0 + 0 * beta_1 + 6 * beta_2 = 7
#   1 * beta_0 + 0 * beta_1 + 8 * beta_2 = 6
#   1 * beta_0 + 2 * beta_1 + 2 * beta_2 = 0
#   1 * beta_0 + 3 * beta_1 + 4 * beta_2 = 0


#   [[1, 0, 0],                   [[1],
#    [1, 1, 0],                    [2],
#    [1, 2, 0],                    [4],
#    [1, 4, 0],    [[beta_0],      [8],
#    [1, 6, 0],     [beta_1],  =   [9],
#    [1, 0, 2],     [beta_2]]      [2],
#    [1, 0, 4],                    [5],
#    [1, 0, 6],                    [7],
#    [1, 0, 8]]                    [6]]
#    [1, 2, 2]]                    [0]]
#    [1, 3, 4]]                    [0]]

# Question 1
#   rating = beta_0 + beta_1 * (slices of beef) + beta_2 * (tbsp peanut butter)


X = Matrix([[1, 0, 0], [1, 1, 0], [1, 2, 0], [1, 4, 0], [
           1, 6, 0], [1, 0, 2], [1, 0, 4], [1, 0, 6], [1, 0, 8], [1, 2, 2], [1, 3, 4]])
y = Matrix([[1], [2], [4], [8], [9], [2], [5], [7], [6], [0], [0]])
betas = (X.transpose() @ X).inverse() @ (X.transpose() @ y)
betas = betas.transpose().elements[0]
print("Betas 0, 1, and 2 respectively:", betas)


def dot(arr1, arr2):
    return sum(x * y for x, y in zip(arr1, arr2))


print("Predicted Rating of 5 slices of beef and no peanut butter:",
      dot([1, 5, 0], betas))

# Question 2
print("Predicted Rating of 5 slices of roast beef and 5 scoops of peanut butter:",
      dot([1, 5, 5], betas))
# 8.02507331378299

# Question 3:
# This prediction is much lower than the prediction from last assignment, because of the extra
# data included with both roast beef and peanut butter on them, making it a little more trustworthy.
# It could be a better prediction though,
# because the ratings say that this sandwich should have a 0 rating.


# QUESTION 4
# Fill out the table with the additional interaction term:

# (slices beef) | (tbsp peanut butter) | (slices beef)(tbsp peanut butter) | Rating |
# -----------------------------------------------------------------------------------
#       0       |           0          |                 0                 |    1   |
#       1       |           0          |                 0                 |    2   |
#       2       |           0          |                 0                 |    4   |
#       4       |           0          |                 0                 |    8   |
#       6       |           0          |                 0                 |    9   |
#       0       |           2          |                 0                 |    2   |
#       0       |           4          |                 0                 |    5   |
#       0       |           6          |                 0                 |    7   |
#       0       |           8          |                 0                 |    6   |
#       2       |           2          |                 4                 |    0   | (new data)
#       3       |           4          |                 12                |    0   | (new data)

# QUESTION 5
# What is the system of equations?

#   1 * beta_0 + 0 * beta_1 + 0 * beta_2 + 0 * beta_3 = 1
#   1 * beta_0 + 1 * beta_1 + 0 * beta_2 + 0 * beta_3 = 2
#   1 * beta_0 + 2 * beta_1 + 0 * beta_2 + 0 * beta_3 = 4
#   1 * beta_0 + 4 * beta_1 + 0 * beta_2 + 0 * beta_3 = 8
#   1 * beta_0 + 6 * beta_1 + 0 * beta_2 + 0 * beta_3 = 9
#   1 * beta_0 + 0 * beta_1 + 2 * beta_2 + 0 * beta_3 = 2
#   1 * beta_0 + 0 * beta_1 + 4 * beta_2 + 0 * beta_3 = 5
#   1 * beta_0 + 0 * beta_1 + 6 * beta_2 + 0 * beta_3 = 7
#   1 * beta_0 + 0 * beta_1 + 8 * beta_2 + 0 * beta_3 = 6
#   1 * beta_0 + 2 * beta_1 + 2 * beta_2 + 4 * beta_3 = 0
#   1 * beta_0 + 3 * beta_1 + 4 * beta_2 + 12 * beta_3 = 0

# QUESTION 6
# What is the matrix equation?

#   [[1, 0, 0, 0 ],                   [[1],
#    [1, 1, 0, 0 ],                    [2],
#    [1, 2, 0, 0 ],                    [4],
#    [1, 4, 0, 0 ],    [[beta_0],      [8],
#    [1, 6, 0, 0 ],     [beta_1],  =   [9],
#    [1, 0, 2, 0 ],     [beta_2],      [2],
#    [1, 0, 4, 0 ],     [beta_3]]      [5],
#    [1, 0, 6, 0 ],                    [7],
#    [1, 0, 8, 0 ],                    [6],
#    [1, 2, 2, 4 ],                    [0],
#    [1, 3, 4, 12]]                    [0]]

# QUESTION 7
# What is the model?

#    rating = beta_0 + beta_1 * (slices beef) + beta_2 * (tbsp peanut butter) + beta_3 * (slices beef)(tbsp peanut butter)

X = Matrix([[1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 2, 0, 0],
            [1, 4, 0, 0],
            [1, 6, 0, 0],
            [1, 0, 2, 0],
            [1, 0, 4, 0],
            [1, 0, 6, 0],
            [1, 0, 8, 0],
            [1, 2, 2, 4],
            [1, 3, 4, 12]])
y = Matrix([[1],
            [2],
            [4],
            [8],
            [9],
            [2],
            [5],
            [7],
            [6],
            [0],
            [0]])
betas = (X.transpose() @ X).inverse() @ (X.transpose() @ y)
betas = betas.transpose().elements[0]
print("Betas 0, 1, 2, and 3 respectively (interaction terms):", betas)
# QUESTION 8
# What is the predicted rating of a sandwich with 5 slices of roast beef AND
# 5 tablespoons of peanut butter (on the same sandwich)?

print("Predicted Rating of 5 slices of roast beef and 5 scoops of peanut butter (interaction terms):",
      dot([1, 5, 5, 25], betas))
# -6.986446833832645

# QUESTION 9
# How does this prediction compare to that from the previous assignment? Did
# including interaction term make the prediction trustworthy? Why or why not?

# Including the interaction term make the prediction much more trustworthy, because it actually went into the negatives
# with both the roast beef and penut butter on the sandwich. It was much lower because one of the terms (the beta_3),
# which has the most weight on the rating, is the combination of both the peanut buttor and the roast beef.
