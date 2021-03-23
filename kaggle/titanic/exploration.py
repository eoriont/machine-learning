import sys, os
sys.path.append("analysis/kaggle/titanic")
sys.path.append("src/models")

from parse_line import parse_line
from data_loading import data_types
from dataframe import DataFrame

if __name__ == "__main__":
    path0 = os.path.join(os.getcwd(), 'datasets', 'titanic', "dataset_of_knowns.csv")
    df = DataFrame.from_csv(path0, data_types=data_types, parser=parse_line)\
        .apply_new("Name", "Surname", lambda x: x.split(",")[0][1:])\
        .apply_new("Cabin", "CabinType", lambda x: None if x is None or len(x) == 0 else x.split(" ")[0][0])\
        .apply_new("Cabin", "CabinNumber", lambda x: None if x is None or len(y := x.split(" ")) == 0 or len(y[0]) == 1 else int(y[0][1:]))\
        .apply_new("Ticket", "TicketType", lambda x: None if x is None or len(y := x.split(" ")) == 1 else y[0])\
        .apply_new("Ticket", "TicketNumber", lambda x: None if len(y := x.split(" ")) == 0 or not y[-1].isnumeric() else int(y[-1]))\
        .filter_columns(["PassengerId", "Survived", "Pclass", "Surname", "Sex", "Age", "SibSp", "Parch", "TicketType", "TicketNumber", "Fare", "CabinType", "CabinNumber", "Embarked"])

    dfa = df.group_by("Pclass").aggregate("Survived", "avg").aggregate("Survived", "count").query("SELECT Pclass, avgSurvived, countSurvived")
    dfb = df.group_by("Sex").aggregate("Survived", "avg").aggregate("Survived", "count").query("SELECT Sex, avgSurvived, countSurvived")
    dfc = df.group_by("SibSp").aggregate("Survived", "avg").aggregate("Survived", "count").query("SELECT SibSp, avgSurvived, countSurvived")
    dfd = df.group_by("Parch").aggregate("Survived", "avg").aggregate("Survived", "count").query("SELECT Parch, avgSurvived, countSurvived")
    dfe = df.group_by("CabinType").aggregate("Survived", "avg").aggregate("Survived", "count").query("SELECT CabinType, avgSurvived, countSurvived")
    dff = df.group_by("Embarked").aggregate("Survived", "avg").aggregate("Survived", "count").query("SELECT Embarked, avgSurvived, countSurvived")

    print(dff.to_array())
