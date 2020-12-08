import sys
sys.path.append("src")
try:
    from decision_tree import DecisionTree
    from dataframe import DataFrame
    from otest import do_assert, cstring
except ImportError as e:
    print(e)


df = DataFrame.from_array(
    [[1, 11, 'A'],
     [1, 12, 'A'],
     [2, 11, 'A'],
     [1, 13, 'B'],
     [2, 13, 'B'],
     [3, 13, 'B'],
     [3, 11, 'B']],
    ['x', 'y', 'class']
)
dt = DecisionTree(df)

do_assert("Row indices", dt.root.row_indices,
          [0, 1, 2, 3, 4, 5, 6])  # these are the indices of data points in the root node

do_assert("Class counts", dt.root.class_counts,
          {
              'A': 3,
              'B': 4
          })

do_assert("impurity", round(dt.root.impurity, 3),
          0.490)  # rounded to 3 decimal places

# dt.possible_splits is a dataframe with columns
# ['feature', 'value', 'goodness of split']
# Note: below is rounded to 3 decimal places
do_assert("possible splits", dt.root.possible_splits.round_column('goodness of split', 3).to_array(),
          [['x', 1.5,  0.085],
           ['x', 2.5,  0.147],
           ['y', 11.5, 0.085],
           ['y', 12.5, 0.276]])

do_assert("best split", dt.root.best_split,
          ('y', 12.5))

dt.split()
# now, the decision tree looks like this:

#           (3A, 4B)
#           /      \
# (y < 12.5)(y >= 12.5)
# (3A, 1B)         (3B)

# "low" refers to the "<" child node
# "high" refers to the ">=" child node
do_assert("low row_indices", dt.root.low.row_indices,
          [0, 1, 2, 6])
do_assert("high row indices", dt.root.high.row_indices,
          [3, 4, 5])

do_assert("low impurity", dt.root.low.impurity,
          0.375)
do_assert("high impurity", dt.root.high.impurity,
          0)

do_assert("low possible splits", dt.root.low.possible_splits.round_column('goodness of split', 3).to_array(),

          [['x', 1.5,  0.125],
           ['x', 2.5,  0.375],
           ['y', 11.5, 0.042]])

do_assert("low best split", dt.root.low.best_split,
          ('x', 2.5))

dt.split()
# now, the decision tree looks like this:

#                     (3A, 4B)
#                     /      \
#            (y < 12.5)       (y >= 12.5)
#            (3A, 1B)         (3B)
#          /         \
# (x < 2.5)          (x >= 2.5)
# (3A)               (1B)

do_assert("low low row indices", dt.root.low.low.row_indices,
          [0, 1, 2])
do_assert("low high row indices", dt.root.low.high.row_indices,
          [6])

do_assert("low low impurity", dt.root.low.low.impurity,
          0)
do_assert("low high impurity", dt.root.low.high.impurity,
          0)

# > ==============================================

df = DataFrame.from_array(
    [[1, 11, 'A'],
     [1, 12, 'A'],
     [2, 11, 'A'],
     [1, 13, 'B'],
     [2, 13, 'B'],
     [3, 13, 'B'],
     [3, 11, 'B']],
    ['x', 'y', 'class']
)
dt = DecisionTree(df)

# currently, the decision tree looks like this:

#   (3A, 4B)

dt.split()
# now, the decision tree looks like this:

#           (3A, 4B)
#           /      \
# (y < 12.5)       (y >= 12.5)
# (3A, 1B)         (3B)

dt.split()
# now, the decision tree looks like this:

#                     (3A, 4B)
#                     /      \
#            (y < 12.5)       (y >= 12.5)
#            (3A, 1B)         (3B)
#          /         \
# (x < 2.5)          (x >= 2.5)
# (3A)               (1B)

do_assert("high row indices 2", dt.root.high.row_indices,
          [3, 4, 5])
do_assert("low low row indices 2", dt.root.low.low.row_indices,
          [0, 1, 2])
do_assert("low high row indices 2", dt.root.low.high.row_indices,
          [6])

dt = DecisionTree(df)

# currently, the decision tree looks like this:

#   (3A, 4B)

dt.fit()
# now, the decision tree looks like this:

#                     (3A, 4B)
#                     /      \
#            (y < 12.5)       (y >= 12.5)
#            (3A, 1B)         (3B)
#          /         \
# (x < 2.5)          (x >= 2.5)
# (3A)               (1B)

do_assert("fit row indices", dt.root.high.row_indices,
          [3, 4, 5])
do_assert("fit row indices 2", dt.root.low.low.row_indices,
          [0, 1, 2])
do_assert("fit row indices 3", dt.root.low.high.row_indices,
          [6])

print(cstring("&6 All tests passed!"))
