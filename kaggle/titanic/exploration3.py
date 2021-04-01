import sys, os, math
sys.path.append("analysis/kaggle/titanic")
sys.path.append("src/models")

from parse_line import parse_line
from data_loading import data_types
from dataframe import DataFrame
from linear_regressor import LinearRegressor

path0 = os.path.join(os.getcwd(), 'datasets', 'titanic', "dataset_of_knowns.csv")
df = DataFrame.from_csv(path0, data_types=data_types, parser=parse_line)\
    .apply("Sex", lambda x: 0 if x == "male" else 1)\
    .where(lambda x: x['Age'] is not None)\
    .apply("SibSp", lambda x: 1 if x == 0 else 0)\
    .apply("Parch", lambda x: 1 if x == 0 else 0)\
    .apply_new("Cabin", "CabinType", lambda x: None if x is None or len(x) == 0 else x.split(" ")[0][0])\
    .select(["Sex", "SibSp", "Parch", "CabinType", "Embarked", "Survived"])

def get_accuracy(lr: LinearRegressor, test: DataFrame):
    return len([1 for x in test.to_json() if round(lr.predict(x)) == x["Survived"]])/len(test)

df_train = df.select_rows(range(500))
df_tests = df.select_rows(range(500, len(df)))

# dfb = df_train.select(["Sex", "Survived"])
# lr = LinearRegressor(dfb, "Survived")

dfc = df_train.select(["Sex", "Parch", "Survived"])
lr = LinearRegressor(dfc, "Survived")

print(get_accuracy(lr, df_train.select(["Sex", "Parch", "Survived"])))
print(get_accuracy(lr, df_tests.select(["Sex", "Parch", "Survived"])))
