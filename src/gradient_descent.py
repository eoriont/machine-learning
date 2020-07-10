class GradientDescent:
    def __init__(self, f):
        self.f = f
        arg_count = f.__code__.co_argcount
        self.minimum = tuple(0 for _ in range(arg_count))

    def grid_search(self, *grid_data):
        param_combos = self.cartesian_product(grid_data)
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
        return tuple(g / (2*delta) for g in result)

    def cartesian_product(self, arrays, points=None):
        points = [[]] if points is None else points
        new_points = [p+[consider] for consider in arrays[0] for p in points]
        return self.cartesian_product(arrays[1:], new_points) if len(arrays) > 1 else new_points
