import sys
import random
import math
sys.path.append('src')
from decision_tree_node import DecisionTreeNode
from dataframe import DataFrame


class DecisionTree:
    def __init__(self, split_metric="gini", dependent_variable="class", max_depth=None, training_percentage=1):
        self.max_depth = max_depth
        self.dependent_variable = dependent_variable
        self.split_metric = split_metric
        self.training_percentage = training_percentage

    def split(self):
        return self.root.split(self.split_metric)

    def fit(self, df):
        df = df.append_columns({
                'id': [i for i in range(df.get_length())]
            }).select_rows(
                [random.randint(0, df.get_length()) for _ in range(math.floor(df.get_length()*self.training_percentage))]
            )
        self.root = DecisionTreeNode(df, self.dependent_variable, max_depth=self.max_depth)
        while self.split():
            pass

    def classify(self, obs):
        return self.root.classify(obs)
