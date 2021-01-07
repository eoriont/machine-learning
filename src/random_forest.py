from decision_tree import DecisionTree


class RandomForest:
    def __init__(self, n):
        self.n = n
        self.dts = [DecisionTree(split_metric='random') for i in range(n)]

    def fit(self, df):
        for dt in self.dts:
            dt.fit(df)

    def prediction(self, obs):
        predictions = []
        for dt in self.dts:
            predictions.append(dt.classify(obs))
        amt = {k: predictions.count(k) for k in set(predictions)}
        return max(amt, key=lambda _: amt.get)
