class GradientDescent:
    def __init__(self, f):
        self.f = f
        arg_count = f.__code__.co_argcount
        self.minimum = tuple(0 for _ in range(arg_count))

    def grid_search(self, *grid_data):
        param_combos = [(i, j) for i in grid_data[0]
                        for j in grid_data[1]]
        min_pos = self.minimum
        min_err = float('inf')
        for pos in param_combos:
            err = self.f(*pos)
            if err < min_err:
                min_err = err
                min_pos = pos
        self.minimum = min_pos

    def descend(self, scaling_factor=0.01, delta=0.01, num_steps=50, logging=False):
        old_min = self.minimum
        gradient = self.compute_gradient(delta)
        self.minimum = tuple(beta - scaling_factor *
                             val for beta, val in zip(old_min, gradient))
        if logging:
            print(self.minimum)
        if num_steps > 1:
            self.descend(scaling_factor, delta, num_steps-1, logging)

    def compute_gradient(self, delta):
        result = []
        for i in range(len(self.minimum)):
            altered_guess = list(self.minimum)
            altered_guess[i] += delta
            f_approx = self.f(*altered_guess)
            altered_guess[i] -= 2*delta
            b_approx = self.f(*altered_guess)
            result.append(f_approx - b_approx)
        return [g / (2*delta) for g in result]


def f(x, y):
    return 1 + (x-1)**2 + (y+5)**2


minimizer = GradientDescent(f)
print(minimizer.minimum)
minimizer.grid_search([-4, -2, 0, -2, 4], [-4, -2, 0, -2, 4])
print(minimizer.minimum)
print(minimizer.compute_gradient(delta=0.01))
minimizer.descend(scaling_factor=0.001, delta=0.01, num_steps=4, logging=True)
print(minimizer.minimum)


data = [(0, 1), (1, 2), (2, 4), (3, 10)]


def sum_squared_error(beta_0, beta_1, beta_2):
    squared_errors = []
    for (x, y) in data:
        estimation = beta_0 + beta_1*x + beta_2*(x**2)
        error = estimation - y
        squared_errors.append(error**2)
    return sum(squared_errors)


minimizer = GradientDescent(sum_squared_error)
minimizer.descend(scaling_factor=0.001, delta=0.01,
                  num_steps=100, logging=True)
print(minimizer.minimum)
print(sum_squared_error(*minimizer.minimum))
