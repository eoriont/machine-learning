class LeaveOneOutCrossValidator:
    def __init__(self, knn, df, prediction_column):
        self.knn = knn
        self.df = df
        self.prediction_column = prediction_column

    def accuracy(self):
        trials = []
        for j in range(len(df)):
            df1, entry = self.df.remove_entry(j)
            self.knn.fit(df1, self.prediction_column)
            trials.append(self.knn.classify(entry) ==
                          entry[self.prediction_column])
        return trials.count(True)/len(trials)
