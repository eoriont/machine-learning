import sys
import os
sys.path.append("src")
try:
    from dataframe import DataFrame
    from random_forest import RandomForest
    from decision_tree import DecisionTree
except ImportError as e:
    print(e)
    sys.exit()


# Get 20%:80% testing data to training data in tuples
def organize_datasets():
    path = os.path.join(os.getcwd(), 'datasets', 'freshman_lbs.csv')
    df = DataFrame.from_csv(path, False)
    l = len(df)
    s = set(i for i in range(l))
    data = []
    for i in range(5):
        testing = df.select_rows(range(i*l//5, (i+1)*l//5))
        training = df.select_rows(s - set(range(i*l//5, (i+1)*l//5)))
        data.append((testing, training))
    return data

# Pass in the training data


def fit(data):
    dt_gini = DecisionTree("gini", dependent_variable="Sex")
    dt_gini.fit(data)

    dt_random = DecisionTree("random", dependent_variable="Sex")
    dt_random.fit(data)

    print("Fit decision trees")

    rf1 = RandomForest(10, dependent_variable="Sex")
    rf1.fit(data)

    print("Fit n=10 random forest")

    rf2 = RandomForest(100, dependent_variable="Sex")
    rf2.fit(data)

    print("Fit n=100 random forest")

    rf3 = RandomForest(1000, dependent_variable="Sex")
    rf3.fit(data)

    print("Fit n=1000 random forest")

    return {'dt_gini': dt_gini, 'dt_random': dt_random, 'rf1': rf1, 'rf2': rf2, 'rf3': rf3}


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


accuracies = []
data = organize_datasets()
for training, testing in data:
    models = fit(training)
    accuracies.append(get_accuracy(models, testing))
accuracies = {x: sum(y[x] for y in accuracies)/len(accuracies) for x in accuracies[0].keys()}
print(accuracies)
