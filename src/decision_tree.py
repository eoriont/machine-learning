import sys
sys.path.append('src')
try:
    from dataframe import DataFrame
except ImportError as e:
    print(e)


class DecisionTree:
    def __init__(self, df):
        self.df = df.append_columns(
            {'id': [i for i in range(df.get_length())]})
        self.root = Node(self.df)

    def split(self):
        self.root.split()


class Node:
    def __init__(self, df):
        self.df = df
        self.row_indices = df.get_column('id')
        classes = df.get_column('class')
        cc = {k: classes.count(k) for k in set(classes)}
        self.class_counts = cc
        self.impurity = gini_impurity(df)

        ps_array = sum((get_splits_for(x, df, self.impurity)
                        for x in ['x', 'y']), [])
        self.possible_splits = DataFrame.from_array(
            ps_array, ['feature', 'value', 'goodness of split'])

        if self.possible_splits.get_length() == 0:
            return
        bs = max(self.possible_splits.to_array(), key=lambda x: x[2])
        self.best_split = (bs[0], bs[1])

        self.low = None
        self.high = None

    def split(self):
        # If haven't split yet
        if self.low is None:
            s, df = self.best_split, self.df
            self.low = Node(df.select_rows_where(lambda x: x[s[0]] <= s[1]))
            self.high = Node(df.select_rows_where(lambda x: x[s[0]] > s[1]))
        else:
            self.low.split()
            self.high.split()


def get_splits_for(col, df, impurity):
    ps = []

    # Dataframe of rows in order of column
    ordered = df.order_by(col).remove_duplicates(col)

    # Go through each 2 rows and split down middle
    for i in range(ordered.get_length() - 1):
        split = sum(x[col] for x in ordered.select_rows([i, i+1]).to_json())/2

        # Get low and high and calculate impurity/goodness
        low = df.select_rows_where(lambda x: x[col] <= split)
        high = df.select_rows_where(lambda x: x[col] > split)
        s = low.get_length() * gini_impurity(low) + \
            high.get_length() * gini_impurity(high)
        s = sum(x.get_length() * gini_impurity(x) for x in [low, high])
        goodness = impurity - s/df.get_length()
        ps.append([col, split, goodness])
    return ps


def gini_impurity(df):
    classes = df.get_column('class')
    cc = {k: classes.count(k) for k in set(classes)}
    return sum(cc[k]/len(classes) * (1 - cc[k]/len(classes)) for k in cc)
