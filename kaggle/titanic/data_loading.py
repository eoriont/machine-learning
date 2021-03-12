import sys, os
sys.path.append("analysis/kaggle/titanic")
sys.path.append("src/models")
sys.path.append("test")
from parse_line import parse_line
from dataframe import DataFrame
from otest import do_assert

data_types = {
    "PassengerId": int,
    "Survived": int,
    "Pclass": int,
    "Name": str,
    "Sex": str,
    "Age": float,
    "SibSp": int,
    "Parch": int,
    "Ticket": str,
    "Fare": float,
    "Cabin": str,
    "Embarked": str
}

if __name__ == "__main__":
    path0 = os.path.join(os.getcwd(), 'datasets', 'titanic', "dataset_of_knowns.csv")
    df = DataFrame.from_csv(path0, data_types=data_types, parser=parse_line)

    do_assert("Columns", df.columns, ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"])

    do_assert("to_array", df.to_array()[:5],
    [[1, 0, 3, '"Braund, Mr. Owen Harris"', "male", 22.0, 1, 0, "A/5 21171", 7.25, None, "S"],
    [2, 1, 1, '"Cumings, Mrs. John Bradley (Florence Briggs Thayer)"', "female", 38.0, 1, 0, "PC 17599", 71.2833, "C85", "C"],
    [3, 1, 3, '"Heikkinen, Miss. Laina"', "female", 26.0, 0, 0, "STON/O2. 3101282", 7.925, None, "S"],
    [4, 1, 1, '"Futrelle, Mrs. Jacques Heath (Lily May Peel)"', "female", 35.0, 1, 0, "113803", 53.1, "C123", "S"],
    [5, 0, 3, '"Allen, Mr. William Henry"', "male", 35.0, 0, 0, "373450", 8.05, None, "S"]])
