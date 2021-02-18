from otest import do_assert
import sys
sys.path.append('src/models')
from gradient_descent import GradientDescent


def single_variable_function(x):
    return (x-1)**2


def two_variable_function(x, y):
    return (x-1)**2 + (y-1)**3


def three_variable_function(x, y, z):
    return (x-1)**2 + (y-1)**3 + (z-1)**4


def six_variable_function(x1, x2, x3, x4, x5, x6):
    return (x1-1)**2 + (x2-1)**3 + (x3-1)**4 + x4 + 2*x5 + 3*x6


def round_down(t):
    if type(t) == tuple:
        return tuple(round(x, 10) for x in t)
    elif type(t) in (float, int):
        return round(t, 10)


tests = [
    {'f': single_variable_function, 'gradient_out': (
        -2.0000000000000018,), 'num_inputs': 1, 'descent': (0.0020000000000000018,), 'gs_in': [[0, 0.25, 0.75]], 'gs_out': (0.75,)},
    {'f': two_variable_function, 'gradient_out': (-2.0000000000000018, 3.0001000000000055),
     'num_inputs': 2, 'descent': (0.0020000000000000018, -0.0030001000000000055), 'gs_in': [[0, 0.25, 0.75], [0.9, 1, 1.1]], 'gs_out': (0.75, 0.9)},
    {'f': three_variable_function, 'gradient_out': (-2.0000000000000018, 3.0001000000000055, -4.0004000000000035),
     'num_inputs': 3, 'descent': (0.0020000000000000018, -0.0030001000000000055, 0.004000400000000004), 'gs_in': [[0, 0.25, 0.75], [0.9, 1, 1.1], [0, 1, 2, 3]], 'gs_out': (0.75, 0.9, 1)},
    {'f': six_variable_function, 'gradient_out': (-2.0000000000000018, 3.0001000000000055, -4.0004000000000035, 1.0000000000000009, 2.0000000000000018, 3.0000000000000027),
     'num_inputs': 6, 'descent': (0.0020000000000000018, -0.0030001000000000055, 0.004000400000000004, -0.0010000000000000009, -0.0020000000000000018, -0.0030000000000000027), 'gs_in': [[0, 0.25, 0.75], [0.9, 1, 1.1], [0, 1, 2, 3],
                                                                                                                                                                                          [-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2]], 'gs_out': (0.75, 0.9, 1, -2, -2, -2)}

]


for test in tests:
    i = test['num_inputs']
    minimizer = GradientDescent(test['f'])
    grad = minimizer.compute_gradient(delta=0.01)
    do_assert(f"compute gradient {i} inputs", round_down(
        grad), round_down(test['gradient_out']))
    minimizer.descend(scaling_factor=0.001, delta=0.01,
                      num_steps=1, logging=False)
    do_assert(f"gradient descent {i} inputs", round_down(
        minimizer.minimum), round_down(test['descent']))

    minimizer.grid_search(*test['gs_in'])
    do_assert(f"grid search {i} inputs", round_down(
        minimizer.minimum), round_down(test['gs_out']))

print("All tests passed!")
