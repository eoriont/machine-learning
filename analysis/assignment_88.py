import sys
import os
import random
sys.path.append("src/models")
from dataframe import DataFrame
from decision_tree import DecisionTree

random.seed(0)

# Get 50%:50% testing data to training data in tuples
def organize_datasets():
    path = os.path.join(os.getcwd(), 'datasets', 'freshman_lbs.csv')
    df = DataFrame.from_csv(path, False).remove_columns(["Weight (lbs, Apr)", "BMI (Apr)"])
    l = len(df)
    return df.select_rows(range(l//2+1)), df.select_rows(range(l//2+1, l+1))

# Pass in the training data
def fit(data):
    dt_gini = DecisionTree("gini", dependent_variable="Sex")
    dt_gini.fit(data)
    return dt_gini


# Pass in the testing data
def get_accuracy(model, data):
    misclassified = [i+len(training) for i, row in enumerate(data.to_json()) if model.classify(row) != row["Sex"]]
    print(misclassified)
    return (len(data)-len(misclassified))/len(data)


training, testing = organize_datasets()
model = fit(training)
print(get_accuracy(model, testing))
