import sys
sys.path.append('src')
try:
    from dataframe import DataFrame
    import random
except ImportError as e:
    print(e)


class DecisionTree:
    def __init__(self, split_metric="gini", dependent_variable="class"):
        self.dependent_variable = dependent_variable
        self.split_metric = split_metric

    def split(self):
        return self.root.split(self.split_metric)

    def fit(self, df):
        self.df = df.append_columns(
            {'id': [i for i in range(df.get_length())]})
        self.root = Node(self.df, self.dependent_variable)
        while self.split():
            pass

    def classify(self, obs):
        return self.root.classify(obs)


class Node:
    def __init__(self, df, dependent_variable):
        self.dependent_variable = dependent_variable
        self.df = df
        self.row_indices = df.get_column('id')
        classes = df.get_column(dependent_variable)
        cc = {k: classes.count(k) for k in set(classes)}
        self.class_counts = cc
        self.impurity = gini_impurity(df, dependent_variable)

        ps_array = sum((get_splits_for(x, df, self.impurity, dependent_variable)
                        for x in df.columns if x not in [dependent_variable, 'id']), [])
        self.possible_splits = DataFrame.from_array(
            ps_array, ['feature', 'value', 'goodness of split'])

        self.low = None
        self.high = None

        if self.possible_splits.get_length() == 0:
            return
        bs = max(self.possible_splits.to_array(), key=lambda x: x[2])
        self.best_split = (bs[0], bs[1])

    def classify(self, obs):
        if self.impurity == 0:
            return list(self.class_counts.keys())[0]
        else:
            if obs[self.best_split[0]] < self.best_split[1]:
                return self.low.classify(obs)
            else:
                return self.high.classify(obs)

    def split(self, split_metric):
        # If haven't split yet
        if self.low is None:
            if self.impurity != 0:
                if split_metric == "random":
                    row = self.possible_splits.random_row()
                    s, df = (row[0], row[1]), self.df
                elif split_metric == "gini":
                    s, df = self.best_split, self.df
                self.low = Node(df.select_rows_where(
                    lambda x: x[s[0]] <= s[1]), self.dependent_variable)
                self.high = Node(df.select_rows_where(
                    lambda x: x[s[0]] > s[1]), self.dependent_variable)
                return True
            else:
                return False
        else:
            return self.low.split(split_metric) or self.high.split(split_metric)


def get_splits_for(col, df, impurity, dv):
    ps = []

    # Dataframe of rows in order of column
    ordered = df.order_by(col).remove_duplicates(col)

    # Go through each 2 rows and split down middle
    for i in range(ordered.get_length() - 1):
        split = sum(x[col] for x in ordered.select_rows([i, i+1]).to_json())/2

        # Get low and high and calculate impurity/goodness
        low = df.select_rows_where(lambda x: x[col] <= split)
        high = df.select_rows_where(lambda x: x[col] > split)
        s = low.get_length() * gini_impurity(low, dv) + \
            high.get_length() * gini_impurity(high, dv)
        s = sum(x.get_length() * gini_impurity(x, dv) for x in [low, high])
        goodness = impurity - s/df.get_length()
        ps.append([col, split, goodness])
    return ps


def gini_impurity(df, dv):
    classes = df.get_column(dv)
    cc = {k: classes.count(k) for k in set(classes)}
    return sum(cc[k]/len(classes) * (1 - cc[k]/len(classes)) for k in cc)
