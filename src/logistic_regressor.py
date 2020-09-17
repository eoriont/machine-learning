import math
from matrix import Matrix
from linear_regressor import LinearRegressor


class LogisticRegressor(LinearRegressor):
    def __init__(self, df, prediction_column, max_value):
        self.max_value = max_value
        df = df.apply(prediction_column,
                      lambda x: math.log((self.max_value/x)-1))
        super().__init__(df, prediction_column)

    def predict(self, inputs):
        return self.max_value/(1+math.exp(super().predict(inputs)))
