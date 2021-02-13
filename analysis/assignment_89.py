import sys
import os
sys.path.append("src")
from dataframe import DataFrame
from random_forest import RandomForest
from decision_tree import DecisionTree


# Pass in the training data
def fit(data):
    dt_random = DecisionTree("random", dependent_variable="Sex", max_depth=4, training_percentage=0.3)
    dt_random.fit(data)

    print("Fit decision trees")

    rf1 = RandomForest(10, dependent_variable="Sex", max_depth=4, training_percentage=0.3)
    rf1.fit(data)

    print("Fit n=10 random forest")

    rf2 = RandomForest(100, dependent_variable="Sex", max_depth=4, training_percentage=0.3)
    rf2.fit(data)

    print("Fit n=100 random forest")

    rf3 = RandomForest(1000, dependent_variable="Sex", max_depth=4, training_percentage=0.3)
    rf3.fit(data)

    print("Fit n=1000 random forest")

    rf4 = RandomForest(10000, dependent_variable="Sex", max_depth=4, training_percentage=0.3)
    rf4.fit(data)

    print("Fit n=10000 random forest")

    return {'dt_random': dt_random, 'rf1': rf1, 'rf2': rf2, 'rf3': rf3, 'rf4': rf4}


# Pass in the testing data
def get_accuracy(models, data):
    accuracy = {}
    for name, m in models.items():
        accuracy[name] = 0
        for row in data.to_json():
            if m.classify(row) == row["Sex"]:
                accuracy[name] += 1
        accuracy[name] /= len(data)
    print("Got accuracies")
    return accuracy


path = os.path.join(os.getcwd(), 'datasets', 'freshman_lbs.csv')
df = DataFrame.from_csv(path, False)
l = len(df)
training, testing = df.select_rows(range(l//2)), df.select_rows(range(l//2, l))

models = fit(training)
accuracies = get_accuracy(models, testing)
print(accuracies)
