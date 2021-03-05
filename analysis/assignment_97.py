import sys
import os
sys.path.append("src/models")
from dataframe import DataFrame
from random_forest import RandomForest
from decision_tree import DecisionTree
from linear_regressor import LinearRegressor
from logistic_regressor import LogisticRegressor
from naive_bayes_classifier import NaiveBayesClassifier
from k_nearest_neighbors_classifier import KNearestNeighborsClassifier

f0 = "dataset_of_knowns.csv"
f1 = "unknowns_to_predict.csv"
f2 = "predictions.csv"

path0 = os.path.join(os.getcwd(), 'datasets', 'titanic', f0)
path1 = os.path.join(os.getcwd(), 'datasets', 'titanic', f1)
path2 = os.path.join(os.getcwd(), 'datasets', 'titanic', f2)

df = DataFrame.from_csv(path0)
prediction_df = DataFrame.from_csv(path1)

features = ["Survived", "Sex", "Age", "SibSp", "Parch", "Pclass"]

df = df.filter_columns(
    features
)\
.apply(
    "Sex",
    lambda x: 0 if x == "male" else 1
)\
.select_rows_where(
    lambda x: "" not in x.values()
)\
.append_columns(
    {
        "Constant": [1 for _ in range(len(df))],
    }
)\

prediction_df = prediction_df.apply(
    "Sex",
    lambda x: 0 if x == "male" else 1
)\
.append_columns(
    {
        "Constant": [1 for _ in range(len(df))],
    }
)\

for x in prediction_df.columns:
    prediction_df = prediction_df.apply(x, lambda x: 0 if x == "" else x)

training_df = df.select_rows(
    range(0, len(df), 2)
)

testing_df = df.select_rows(
    range(1, len(df), 2)
)


def get_accuracy(model, testing_df):
    correct = [0 for row in testing_df.to_json() if round(model.predict(row)) == row["Survived"]]
    return len(correct)/len(testing_df)

def get_accuracies(models, testing_df):
    for x in models:
        print(type(x).__name__)
        print("Training Accuracy:")
        print(get_accuracy(x, training_df))
        print("----")
        print("Testing Accuracy:")
        print(get_accuracy(x, testing_df))

        print("=================")

def get_predictions(model, unknown_df):
    predictions = DataFrame({"PassengerId":[], "Survived":[]})
    for row in unknown_df.to_json():
        p = model.predict(row)
        predictions = predictions.add_entry({"PassengerId": row["PassengerId"], "Survived": round(p)})
    return predictions



linear_reg = LinearRegressor(training_df, "Survived")
# logistic_reg = LogisticRegressor(training_df.apply("Survived", lambda x: 0.1 if x == 0 else 0.9), "Survived")
# dt_gini1 = DecisionTree(split_metric="gini", dependent_variable="Survived", max_depth=5)
# dt_gini1.fit(training_df)
# dt_gini2 = DecisionTree(split_metric="gini", dependent_variable="Survived", max_depth=10)
# dt_gini2.fit(training_df)

# rf_1 = RandomForest(100, dependent_variable="Survived", max_depth=3)
# rf_1.fit(training_df)
# rf_2 = RandomForest(100, dependent_variable="Survived", max_depth=5)
# rf_2.fit(training_df)

# nbc_df = df
# for x in nbc_df.columns:
#     mean = sum(nbc_df.get_column(x))/len(nbc_df)
#     nbc_df = nbc_df.apply(x, lambda x: x >= mean)

# nbc = NaiveBayesClassifier(nbc_df, "Survived")

# knn1 = KNearestNeighborsClassifier(5, metric="manhattan")
# knn1.fit(training_df, "Survived")

# knn2 = KNearestNeighborsClassifier(10, metric="manhattan")
# knn2.fit(training_df, "Survived")

# get_accuracies([linear_reg], testing_df)
get_predictions(linear_reg, prediction_df).save_csv(path2)
