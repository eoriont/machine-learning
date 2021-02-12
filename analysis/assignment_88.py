import sys
import os
import random
sys.path.append("src")
try:
    from dataframe import DataFrame
    from random_forest import RandomForest
    from decision_tree import DecisionTree
except ImportError as e:
    print(e)
    sys.exit()

random.seed(0)

# Get 50%:50% testing data to training data in tuples
def organize_datasets():
    path = os.path.join(os.getcwd(), 'datasets', 'freshman_lbs.csv')
    df = DataFrame.from_csv(path, False)
    l = df.get_length()
    return df.select_rows(range(l//2)), df.select_rows(range(l//2, l+1))

# Pass in the training data
def fit(data):
    dt_gini = DecisionTree("gini", dependent_variable="Sex")
    dt_gini.fit(data)
    return dt_gini


# Pass in the testing data
def get_accuracy(model, data):
    # return sum(1 for row in data.to_json() if model.classify(row) == row["Sex"]) / data.get_length()
    accuracy = 0
    for i, row in enumerate(data.to_json()):
        if model.classify(row) == row["Sex"]:
            accuracy += 1
        else:
            print(i + training.get_length())
    accuracy /= data.get_length()
    return accuracy


training, testing = organize_datasets()
model = fit(training)
print(get_accuracy(model, testing))
