import sys
sys.path.append('src')
try:
    from dataframe import DataFrame
    import random
except ImportError as e:
    print(e)


class DecisionTree:
    def __init__(self, split_metric="gini", dependent_variable="class", max_depth=None):
        self.max_depth = max_depth
        self.dependent_variable = dependent_variable
        self.split_metric = split_metric

    def split(self):
        return self.root.split(self.split_metric)

    def fit(self, df):
        self.df = df.append_columns(
            {'id': [i for i in range(df.get_length())]})
        self.root = Node(self.df, self.dependent_variable, max_depth=self.max_depth)
        while self.split():
            pass

    def classify(self, obs):
        return self.root.classify(obs)


class Node:
    def __init__(self, df, dependent_variable, max_depth=None):
        self.max_depth = max_depth
        self.dependent_variable = dependent_variable
        self.df = df
        self.row_indices = df.get_column('id')
        classes = df.get_column(dependent_variable)
        cc = {k: classes.count(k) for k in set(classes)}
        self.class_counts = cc
        self.impurity = gini_impurity(df, dependent_variable)

        self.low = None
        self.high = None

        self.possible_splits = DataFrame.from_array(sum((get_splits_for(x, df, self.impurity, dependent_variable)
                        for x in df.columns if x not in [dependent_variable, 'id']), []), ['col', 'split', 'goodness of split'])

        self.best_split = get_best_split(df, dependent_variable, self.impurity, None)

    def classify(self, obs):
        if self.max_depth is None:
            if self.impurity == 0:
                return max(self.class_counts, key=lambda x: self.class_counts[x])
            else:
                if obs[self.best_split[0]] < self.best_split[1]:
                    return self.low.classify(obs)
                else:
                    return self.high.classify(obs)
        else:
            return max(self.class_counts, key=self.class_counts.get)

    def split(self, split_metric):
        if self.max_depth is None or self.max_depth <= 0:
            # If haven't split yet
            if self.low is None:
                if self.impurity != 0:
                    new_depth = None if self.max_depth is None else self.max_depth-1
                    if split_metric == "random":
                        s = get_random_split_column(self.df, self.dependent_variable, self.impurity)
                        if s is None:
                            #! This is still a hack
                            self.impurity = 0
                            return False
                        self.best_split = s
                    elif split_metric == "gini":
                        s = self.best_split
                    self.low = Node(self.df.select_rows_where(
                        lambda x: x[s[0]] <= s[1]), self.dependent_variable, max_depth=new_depth)
                    self.high = Node(self.df.select_rows_where(
                        lambda x: x[s[0]] > s[1]), self.dependent_variable, max_depth=new_depth)
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

def get_best_split(df, dependent_variable, impurity, col):
    if col is None:
        ps_array = sum((get_splits_for(x, df, impurity, dependent_variable)
                        for x in df.columns if x not in [dependent_variable, 'id']), [])
    else:
        ps_array = get_splits_for(col, df, impurity, dependent_variable)
    possible_splits = DataFrame.from_array(
        ps_array,
        ['feature', 'value', 'goodness of split'])
    if possible_splits.get_length() == 0:
        return
    bs = max(possible_splits.to_array(), key=lambda x: x[2])
    return bs

def get_random_split_column(df, dv, impurity):
    visited = []
    while len(visited) < len(df.columns)-2:
        col = random.choice([i for i in df.columns if i not in ["id", dv]+visited])
        s = get_best_split(df, dv, impurity, col)
        if s is not None:
            return s
        visited += col
    return None
