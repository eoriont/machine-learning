import sys
import os
sys.path.append("src/models")
from dataframe import DataFrame
from logistic_regressor import LogisticRegressor
import matplotlib.pyplot as plt
plt.style.use("bmh")


#> Part A

df = DataFrame.from_array(
    [[1,0],
    [2,0],
    [3,0],
    [2,1],
    [3,1],
    [4,1]],
    columns = ['x', 'y'])


offsets = [0.1, 0.01, 0.001, 0.0001]

def trim(offset):
    def _trim(x):
        if x == 0:
            return offset
        elif x == 1:
            return 1-offset
        else:
            return x
    return _trim

for o in offsets:
    df1 = df.apply("y", trim(o))
    lr = LogisticRegressor(df1, "y", max_value=1)
    x_vals = [i for i in range(100)]
    y_vals = [lr.predict({"x": i}) for i in x_vals]
    plt.plot(x_vals, y_vals)

plt.legend(offsets)
plt.savefig("assignment_109a.png")

#> Part B

plt.clf()
df = DataFrame.from_array(
[[1,0],
[2,0],
[3,0],
[2,1],
[3,1],
[4,1]],
columns = ['x', 'y'])
df = df.apply_new("x", "constant", lambda _: 1)

reg = LogisticRegressor(df, dependent_variable='y')

reg.set_coefficients({'constant': 0.5, 'x': 0.5})

alpha = 0.01
delta = 0.01
num_steps = 20000
# reg.gradient_descent(alpha, delta, num_steps, debug_mode=False)

# print(reg.coefficients)
reg.set_coefficients({'constant': 2.7911, 'x': -1.1165})
# {'constant': 2.7911, 'x': -1.1165}

x_vals = [i for i in range(100)]
y_vals = [reg.predict({"x": i, "constant": 1}) for i in x_vals]
plt.plot(x_vals, y_vals)

plt.legend(["Gradient Descent"])
plt.title("Gradient Descent Plot")
plt.savefig("assignment_109b.png")
