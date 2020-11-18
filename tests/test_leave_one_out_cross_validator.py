import sys
from otest import do_assert
sys.path.append("src")
try:
    from k_nearest_neighbors_classifier import KNearestNeighborsClassifier
    from leave_one_out_cross_validator import LeaveOneOutCrossValidator
    from dataframe import DataFrame
except ImportError as e:
    print(e)

columns = ['Cookie Type', 'Portion Eggs',
           'Portion Butter', 'Portion Sugar', 'Portion Flour']
data = [['Shortbread',     0.14,       0.14,      0.28,     0.44],
        ['Shortbread',     0.10,       0.18,      0.28,     0.44],
        ['Shortbread',     0.12,       0.10,      0.33,     0.45],
        ['Shortbread',     0.10,       0.25,      0.25,     0.40],
        ['Sugar',     0.00,       0.10,      0.40,     0.50],
        ['Sugar',     0.00,       0.20,      0.40,     0.40],
        ['Sugar',     0.02,       0.08,      0.45,     0.45],
        ['Sugar',     0.10,       0.15,      0.35,     0.40],
        ['Sugar',     0.10,       0.08,      0.35,     0.47],
        ['Sugar',     0.00,       0.05,      0.30,     0.65],
        ['Fortune',     0.20,       0.00,      0.40,     0.40],
        ['Fortune',     0.25,       0.10,      0.30,     0.35],
        ['Fortune',     0.22,       0.15,      0.50,     0.13],
        ['Fortune',     0.15,       0.20,      0.35,     0.30],
        ['Fortune',     0.22,       0.00,      0.40,     0.38],
        ['Shortbread',     0.05,       0.12,      0.28,     0.55],
        ['Shortbread',     0.14,       0.27,      0.31,     0.28],
        ['Shortbread',     0.15,       0.23,      0.30,     0.32],
        ['Shortbread',     0.20,       0.10,      0.30,     0.40]]

df = DataFrame.from_array(data, columns)
json_data = df.to_json()
knn = KNearestNeighborsClassifier(df, prediction_column='Cookie Type')
def classifier(observation): return knn.classify(observation, 5)


cv = LeaveOneOutCrossValidator(classifier, json_data)
do_assert("accuracy", cv.accuracy(),
          0.6842105263157895)

# Note: the following is included to help you debug.
# Row 0 -- True Class is Shortbread; Predicted Class was Shortbread
# Row 1 -- True Class is Shortbread; Predicted Class was Shortbread
# Row 2 -- True Class is Shortbread; Predicted Class was Shortbread
# Row 3 -- True Class is Shortbread; Predicted Class was Shortbread
# Row 4 -- True Class is Sugar; Predicted Class was Sugar
# Row 5 -- True Class is Sugar; Predicted Class was Sugar
# Row 6 -- True Class is Sugar; Predicted Class was Sugar
# Row 7 -- True Class is Sugar; Predicted Class was Shortbread
# Row 8 -- True Class is Sugar; Predicted Class was Shortbread
# Row 9 -- True Class is Sugar; Predicted Class was Sugar
# Row 10 -- True Class is Fortune; Predicted Class was Shortbread
# Row 11 -- True Class is Fortune; Predicted Class was Fortune
# Row 12 -- True Class is Fortune; Predicted Class was Fortune
# Row 13 -- True Class is Fortune; Predicted Class was Shortbread
# Row 14 -- True Class is Fortune; Predicted Class was Shortbread
# Row 15 -- True Class is Shortbread; Predicted Class was Sugar
# Row 16 -- True Class is Shortbread; Predicted Class was Shortbread
# Row 17 -- True Class is Shortbread; Predicted Class was Shortbread
# Row 18 -- True Class is Shortbread; Predicted Class was Shortbread

accuracies = []
for k in range(1, len(data)-1):
    def classifier(observation): return knn.classify(observation, k)
    cv = LeaveOneOutCrossValidator(classifier, json_data)
    accuracies.append(cv.accuracy())

do_assert("accuracies2", accuracies,
          [0.5789473684210527,
           0.5263157894736842,
           0.5789473684210527,
           0.5789473684210527,
           0.6842105263157895,
           0.6842105263157895,
           0.5789473684210527,
           0.631578947368421,
           0.5789473684210527,
           0.5263157894736842,
           0.5263157894736842,
           0.42105263157894735,
           0.47368421052631576,
           0.42105263157894735,
           0.42105263157894735,
           0.3157894736842105,
           0.42105263157894735])
