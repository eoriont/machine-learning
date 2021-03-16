from decision_tree import DecisionTree
from dataframe import DataFrame

class RandomForest:
    def __init__(self, n, dependent_variable="class", max_depth=None, training_percentage=1):
        self.dependent_variable = dependent_variable
        self.n = n
        self.max_depth = max_depth
        self.dts = [DecisionTree(
                split_metric='random',
                dependent_variable=dependent_variable,
                max_depth=max_depth,
                training_percentage=training_percentage
            ) for i in range(n)]

    def fit(self, df):
        for dt in self.dts:
            dt.fit(df)

    def classify(self, obs):
        predictions = []
        for dt in self.dts:
            predictions.append(dt.classify(obs))
        amt = {k: predictions.count(k) for k in set(predictions)}
        return max(amt, key=lambda x: amt[x])

    def predict(self, obs):
        return self.classify(obs)
