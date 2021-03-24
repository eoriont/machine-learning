import sys, os, math
sys.path.append("analysis/kaggle/titanic")
sys.path.append("src/models")

from parse_line import parse_line
from data_loading import data_types
from dataframe import DataFrame

if __name__ == "__main__":
    path0 = os.path.join(os.getcwd(), 'datasets', 'titanic', "dataset_of_knowns.csv")
    df = DataFrame.from_csv(path0, data_types=data_types, parser=parse_line)\
        .apply_new("Age", "AgeGroup", lambda x: None if x is None else x//10)\
        .where(lambda x: x['Age'] is not None)\
        .apply_new("Fare", "FareGroup", lambda x: None if x is None or x//5 == 0 else int(math.log(x//5, 2)))\
        .where(lambda x: x['FareGroup'] is not None)

    dfa = df.group_by("AgeGroup").aggregate("Survived", "avg").aggregate("Survived", "count").select(["AgeGroup", "avgSurvived", "countSurvived"])
    dfb = df.group_by("FareGroup").aggregate("Survived", "avg").aggregate("Survived", "count").select(["FareGroup", "avgSurvived", "countSurvived"])

    print(dfb.to_array())
