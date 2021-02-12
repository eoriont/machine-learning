from dataframe import DataFrame
import random

class DecisionTreeNode:
    def __init__(self, df, dependent_variable, max_depth=None):
        self.max_depth = max_depth
        self.dependent_variable = dependent_variable
        self.df = df
        self.row_indices = df.get_column('id')
        classes = df.get_column(dependent_variable)
        cc = {k: classes.count(k) for k in set(classes)}
        self.class_counts = cc
        self.impurity = self.gini_impurity(df)

        self.low = None
        self.high = None

        self.possible_splits = DataFrame.from_array(sum((self.get_splits_for(x)
                                                        for x in df.columns
                                                        if x not in [dependent_variable, 'id']), []),
                        ['col', 'split', 'goodness of split'])

        self.best_split = self.get_best_split(None)

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

    # Splits the node into 2 children nodes, returns if it has done so
    def split(self, split_metric):
        if self.max_depth is None or self.max_depth <= 0:
            # If haven't split yet
            if self.low is None:
                if self.impurity != 0:
                    new_depth = None if self.max_depth is None else self.max_depth-1

                    # Set s to column to split
                    if split_metric == "random":
                        s = self.get_random_split_column()
                        if s is None:
                            #! This is still a hack
                            self.impurity = 0
                            return False
                        self.best_split = s
                    elif split_metric == "gini":
                        s = self.best_split

                    # Make high and low nodes based on split
                    self.low = DecisionTreeNode(self.df.select_rows_where(
                        lambda x: x[s[0]] <= s[1]), self.dependent_variable, max_depth=new_depth)
                    self.high = DecisionTreeNode(self.df.select_rows_where(
                        lambda x: x[s[0]] > s[1]), self.dependent_variable, max_depth=new_depth)

                    # Return true for 'has split'
                    return True
                else:
                    # Return false for 'has not split'
                    return False
            else:
                # If already split, return if any children split
                return self.low.split(split_metric) or self.high.split(split_metric)

    # Returns the gini impurity
    def gini_impurity(self, df):
        classes = df.get_column(self.dependent_variable)
        cc = {k: classes.count(k) for k in set(classes)}
        return sum(cc[k]/len(classes) * (1 - cc[k]/len(classes)) for k in cc)

    def get_splits_for(self, col):
        # Dataframe of rows in order of column
        ordered = self.df.order_by(col).remove_duplicates(col)

        # Go through each 2 rows and split down middle
        ps = []
        for i in range(ordered.get_length() - 1):
            split = sum(x[col] for x in ordered.select_rows([i, i+1]).to_json())/2

            # Get low and high
            low = self.df.select_rows_where(lambda x: x[col] <= split)
            high = self.df.select_rows_where(lambda x: x[col] > split)

            # Calculate goodness
            s = sum(x.get_length() * self.gini_impurity(x) for x in [low, high])
            goodness = self.impurity - s/self.df.get_length()
            ps.append([col, split, goodness])
        return ps


    def get_best_split(self, col):
        if col is None:
            # Get possible splits for all columns
            ps_array = sum((self.get_splits_for(x)
                            for x in self.df.columns
                            if x not in [self.dependent_variable, 'id']), [])
        else:
            # Get possible splits for col
            ps_array = self.get_splits_for(col)
        possible_splits = DataFrame.from_array(
            ps_array,
            ['feature', 'value', 'goodness of split'])
        if possible_splits.get_length() == 0:
            return

        # Return split with best 'goodness of split'
        bs = max(possible_splits.to_array(), key=lambda x: x[2])
        return bs

    def get_random_split_column(self):
        # Get best splits for all columns
        best_splits = [bs
                        for col in self.df.columns
                        if col not in ["id", self.dependent_variable]
                        if (bs := self.get_best_split(col))]
        if len(best_splits) == 0:
            return None
        return random.choice(best_splits)
