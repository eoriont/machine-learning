from matrix import Matrix

class LinearRegressor:
    def __init__(self, df, prediction_column):
        self.pc = prediction_column
        self.prediction_column = df.data_dict[self.pc]
        self.df = df.remove_columns([prediction_column])
        self.coefficients = self.solve_coefficients()

    def solve_coefficients(self):
        x_data = [list(t) for t in zip(*self.df.to_array())]
        y_data = [self.prediction_column]
        X, Y = ~Matrix(x_data), ~Matrix(y_data)
        result = (~X @ X).inverse() @ (~X @ Y)
        return {key: val for key, val in zip(self.df.columns, result[:, 0])}

    def gather_all_inputs(self, terms):
        to_calc = [x.split("_") for x in self.df.columns if '_' in x]
        res = {'_'.join([x, y]): terms[x]*terms[y] for x, y in to_calc}
        res.update(terms)
        return res

    def predict(self, inputs):
        full_inputs = self.gather_all_inputs(inputs)
        zipped = [(v, self.coefficients[k]) for k, v in full_inputs.items()
                                            if k in self.df.columns and k != self.prediction_column]
        return sum(x*y for x, y in zipped)

    def round_coefficients(self):
        self.coefficients = {key: round(val, 7)
                             for key, val in self.coefficients.items()}

    def rss(self, df):
        df = df.remove_columns(['y'])
        return sum((self.predict(df.to_entry(x)) - y)**2 for x, y in zip(df.to_array(), self.prediction_column))
