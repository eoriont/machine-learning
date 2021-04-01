import math
from linear_regressor import LinearRegressor
from dataframe import DataFrame
import matplotlib.pyplot as plt
plt.style.use("bmh")


class LogisticRegressor(LinearRegressor):
    def __init__(self, df, dependent_variable, max_value=1):
        self.max_value = max_value
        df = df.apply(dependent_variable, self.transform)
        super().__init__(df, dependent_variable)

    def transform(self, x):
        return math.log(self.max_value/x - 1)

    def predict(self, inputs):
        return self.max_value/(1+math.exp(super().predict(inputs)))

    def calc_rss(self):
        return sum((self.predict(self.df.to_entry(x)) - y)**2 for x, y in zip(self.df.to_array(), self.prediction_column))

    def set_coefficients(self, coeffs):
        self.coefficients = coeffs

    def calc_gradient(self, delta):
        grad = {}
        for col in self.df.columns:
            self.coefficients[col] += delta
            f_approx = self.calc_rss()
            self.coefficients[col] -= 2*delta
            b_approx = self.calc_rss()
            self.coefficients[col] += delta
            grad[col] = (f_approx - b_approx)/(2*delta)
        return grad

    def gradient_descent(self, alpha, delta, num_steps, debug_mode=False):
        for i in range(num_steps):
            grad = self.calc_gradient(delta)
            rss = self.calc_rss()
            if debug_mode:
                print("step", i)
                print("\tgradient=", grad)
                print("\tcoefficients=", self.coefficients)
                print("\trss=", rss)
                print("")
                input()
            for col, val in grad.items():
                self.coefficients[col] -= alpha * val
