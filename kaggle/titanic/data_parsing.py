import sys, os
sys.path.append("analysis/kaggle/titanic")
sys.path.append("src/models")
sys.path.append("test")

from parse_line import parse_line
from data_loading import data_types
from dataframe import DataFrame
from otest import do_assert

if __name__ == "__main__":
    path0 = os.path.join(os.getcwd(), 'datasets', 'titanic', "dataset_of_knowns.csv")
    df = DataFrame.from_csv(path0, data_types=data_types, parser=parse_line)\
        .apply_new("Name", "Surname", lambda x: x.split(",")[0][1:])\
        .apply_new("Cabin", "CabinType", lambda x: None if x is None or len(x) == 0 else x.split(" ")[0][0])\
        .apply_new("Cabin", "CabinNumber", lambda x: None if x is None or len(y := x.split(" ")) == 0 or len(y[0]) == 1 else int(y[0][1:]))\
        .apply_new("Ticket", "TicketType", lambda x: None if x is None or len(y := x.split(" ")) == 1 else y[0])\
        .apply_new("Ticket", "TicketNumber", lambda x: None if len(y := x.split(" ")) == 0 or not y[-1].isnumeric() else int(y[-1]))\
        .filter_columns(["PassengerId", "Survived", "Pclass", "Surname", "Sex", "Age", "SibSp", "Parch", "TicketType", "TicketNumber", "Fare", "CabinType", "CabinNumber", "Embarked"])

    do_assert("correct dataframe processing", df.to_array()[:5],
    [[1, 0, 3, "Braund", "male", 22.0, 1, 0, "A/5", 21171, 7.25, None, None, "S"],
    [2, 1, 1, "Cumings", "female", 38.0, 1, 0, "PC", 17599, 71.2833, "C", 85, "C"],
    [3, 1, 3, "Heikkinen", "female", 26.0, 0, 0, "STON/O2.", 3101282, 7.925, None, None, "S"],
    [4, 1, 1, "Futrelle", "female", 35.0, 1, 0, None, 113803, 53.1, "C", 123, "S"],
    [5, 0, 3, "Allen", "male", 35.0, 0, 0, None, 373450, 8.05, None, None, "S"]])
