import math
from matrix import Matrix
from linear_regressor import LinearRegressor


class LogisticRegressor(LinearRegressor):
    def __init__(self, df, prediction_column, max_value=1):
        self.max_value = max_value
        df = df.trim_column(prediction_column, max_value)
        df = df.apply(prediction_column, self.transform)
        super().__init__(df, prediction_column)

    def transform(self, x):
        return math.log(self.max_value/x - 1)

    def predict(self, inputs):
        return self.max_value/(1+math.exp(super().predict(inputs)))
