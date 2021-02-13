import math


class NaiveBayesClassifier:
    def __init__(self, df, dependent_variable):
        self.df = df
        self.dependent_variable = dependent_variable

    def probability(self, col, val):
        return self.df.get_column(col).count(val) / len(self.df.get_column(col))

    def conditional_probability(self, criteria, given):
        given_arr = self.df.select_rows_where(
            lambda x: x[given[0]] == given[1])
        cprob = len(given_arr.select_rows_where(
            lambda x: x[criteria[0]] == criteria[1]))/len(given_arr)
        return cprob

    def likelihood(self, given, observed):
        return math.prod(self.conditional_probability(x, given) for x in observed.items()) * self.probability(*given)

    # I'd like to change this garbage but we don't pass in the column
    def classify(self, observed):
        to_classify = (set(self.df.columns) - set(observed.keys())).pop()
        l = {x: self.likelihood((to_classify, x), observed)
             for x in [True, False]}
        return to_classify, max(l, key=l.get)
