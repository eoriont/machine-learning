from decision_tree import DecisionTree
from dataframe import DataFrame


class RandomForest:
    def __init__(self, n, dependent_variable="class"):
        self.dependent_variable = dependent_variable
        self.n = n
        self.dts = [DecisionTree(
            split_metric='random', dependent_variable=dependent_variable) for i in range(n)]

    def fit(self, df):
        for dt in self.dts:
            dt.fit(df)

    def classify(self, obs):
        predictions = []
        for dt in self.dts:
            predictions.append(dt.classify(obs))
        amt = {k: predictions.count(k) for k in set(predictions)}
        return max(amt, key=lambda x: amt[x])


if __name__ == "__main__":
    rf = RandomForest(5)
    data = [[2, 13, 'B'], [2, 13, 'B'], [2, 13, 'B'], [2, 13, 'B'], [2, 13, 'B'], [2, 13, 'B'],
            [3, 13, 'B'], [3, 13, 'B'], [3, 13, 'B'], [
                3, 13, 'B'], [3, 13, 'B'], [3, 13, 'B'],
            [2, 12, 'B'], [2, 12, 'B'],
            [3, 12, 'A'], [3, 12, 'A'],
            [3, 11, 'A'], [3, 11, 'A'],
            [3, 11.5, 'A'], [3, 11.5, 'A'],
            [4, 11, 'A'], [4, 11, 'A'],
            [4, 11.5, 'A'], [4, 11.5, 'A'],
            [2, 10.5, 'A'], [2, 10.5, 'A'],
            [3, 10.5, 'B'],
            [4, 10.5, 'A']]
    df = DataFrame.from_array(data, ['x', 'y', 'class'])
    rf.fit(df)
    # According to the decision_tree test file, this should be A
    print(rf.classify({'x': 3.75, 'y': 10.5}))
