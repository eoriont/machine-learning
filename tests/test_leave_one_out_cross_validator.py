import sys
from otest import do_assert
sys.path.append("src/models")
from k_nearest_neighbors_classifier import KNearestNeighborsClassifier
from leave_one_out_cross_validator import LeaveOneOutCrossValidator
from dataframe import DataFrame

df = DataFrame.from_array(
    [['Shortbread',     0.14,       0.14,      0.28,     0.44],
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
        ['Shortbread',     0.20,       0.10,      0.30,     0.40]],
    ['Cookie Type', 'Portion Eggs', 'Portion Butter',
        'Portion Sugar', 'Portion Flour']
)
knn = KNearestNeighborsClassifier(k=5)

cv = LeaveOneOutCrossValidator(knn, df, prediction_column='Cookie Type')
do_assert("accuracy", cv.accuracy(),
          0.7894736842105263)

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
# Row 10 -- True Class is Fortune; Predicted Class was Fortune (Updated!)
# Row 11 -- True Class is Fortune; Predicted Class was Fortune
# Row 12 -- True Class is Fortune; Predicted Class was Fortune
# Row 13 -- True Class is Fortune; Predicted Class was Shortbread
# Row 14 -- True Class is Fortune; Predicted Class was Fortune (Updated!)
# Row 15 -- True Class is Shortbread; Predicted Class was Sugar
# Row 16 -- True Class is Shortbread; Predicted Class was Shortbread
# Row 17 -- True Class is Shortbread; Predicted Class was Shortbread
# Row 18 -- True Class is Shortbread; Predicted Class was Shortbread

accuracies = []
for k in range(1, len(df)-1):
    knn = KNearestNeighborsClassifier(k)
    cv = LeaveOneOutCrossValidator(knn, df, prediction_column='Cookie Type')
    accuracies.append(cv.accuracy())

do_assert("accuracies2", accuracies,
          [0.5789473684210527,
           0.5789473684210527,  # (Updated!)
           0.5789473684210527,
           0.5789473684210527,
           0.7894736842105263,  # (Updated!)
           0.6842105263157895,
           0.5789473684210527,
           0.5789473684210527,  # (Updated!)
           0.6842105263157895,  # (Updated!)
           0.5263157894736842,
           0.47368421052631576,  # (Updated!)
           0.42105263157894735,
           0.42105263157894735,  # (Updated!)
           0.3684210526315789,  # (Updated!)
           0.3684210526315789,  # (Updated!)
           0.3684210526315789,  # (Updated!)
           0.42105263157894735])
