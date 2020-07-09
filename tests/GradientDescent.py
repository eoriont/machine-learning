from src.GradientDescent import GradientDescent


def single_variable_function(x):
    return (x-1)**2


def two_variable_function(x, y):
    return (x-1)**2 + (y-1)**3


def three_variable_function(x, y, z):
    return (x-1)**2 + (y-1)**3 + (z-1)**4


def six_variable_function(x1, x2, x3, x4, x5, x6):
    return (x1-1)**2 + (x2-1)**3 + (x3-1)**4 + x4 + 2*x5 + 3*x6


tests = [
    {'f': single_variable_function, 'gradient_out': (
        -2.0000000000000018,), 'num_inputs': 1, 'descent': (0.0020000000000000018,)},
    {'f': two_variable_function, 'gradient_out': (-2.0000000000000018, 3.0001000000000055),
     'num_inputs': 2, 'descent': (0.0020000000000000018, -0.0030001000000000055)},
    {'f': three_variable_function, 'gradient_out': (-2.0000000000000018, 3.0001000000000055, -4.0004000000000035),
     'num_inputs': 3, 'descent': (0.0020000000000000018, -0.0030001000000000055, 0.004000400000000004)},
    {'f': six_variable_function, 'gradient_out': (-2.0000000000000018, 3.0001000000000055, -4.0004000000000035, 1.0000000000000009, 2.0000000000000018, 3.0000000000000027),
     'num_inputs': 6, 'descent': (0.0020000000000000018, -0.0030001000000000055, 0.004000400000000004, -0.0010000000000000009, -0.0020000000000000018, -0.0030000000000000027)}

]

for test in tests:
    minimizer = GradientDescent(test['f'])
    grad = minimizer.compute_gradient(delta=0.01)
    assert grad == test['gradient_out'], f"Compute Gradient broke on {test['num_inputs']} inputs " + str(
        grad)
    minimizer.descend(scaling_factor=0.001, delta=0.01,
                      num_steps=1, logging=False)
    assert minimizer.minimum == test['descent'], f"Gradient Descent broke on {test['num_inputs']} inputs' " + str(
        minimizer.minimum)

print("All tests passed!")
