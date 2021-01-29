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
dt = DecisionTree()
dt.fit(df)

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

do_assert("best split", tuple(dt.root.best_split[:2]),
          ('y', 12.5))

# dt.split()
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

do_assert("low best split", tuple(dt.root.low.best_split[:2]),
          ('x', 2.5))

# dt.split()
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
dt = DecisionTree()
dt.fit(df)

# currently, the decision tree looks like this:

#   (3A, 4B)

# dt.split()
# now, the decision tree looks like this:

#           (3A, 4B)
#           /      \
# (y < 12.5)       (y >= 12.5)
# (3A, 1B)         (3B)

# dt.split()
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

dt = DecisionTree()
dt.fit(df)

# currently, the decision tree looks like this:

#   (3A, 4B)

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

# > ===============================================================

data = [[2, 13, 'B'], [2, 13, 'B'], [2, 13, 'B'], [2, 13, 'B'], [2, 13, 'B'], [2, 13, 'B'],
        [3, 13, 'B'], [3, 13, 'B'], [3, 13, 'B'], [
            3, 13, 'B'], [3, 13, 'B'], [3, 13, 'B'],
        [2, 12, 'B'], [2, 12, 'B'],
        [3, 12, 'A'], [3, 12, 'A'],
        [3, 11, 'A'], [3, 11, 'A'],
        [3, 11.5, 'A'], [3, 11.5, 'A'],
        [4, 11, 'A'], [4, 11, 'A'],
        [4, 11.5, 'A'], [4, 11.5, 'A'],
        [2, 10.5, 'A'], [2, 10.5, 'A'],
        [3, 10.5, 'B'],
        [4, 10.5, 'A']]
df = DataFrame.from_array(data, ['x', 'y', 'class'])
dt = DecisionTree()
dt.fit(df)

# The tree should look like this:

#                          (13A, 15B)
#                           /      \
#                 (y < 12.5)       (y >= 12.5)
#                 (13A, 3B)        (12B)
#                 /         \
#         (x < 2.5)          (x >= 2.5)
#         (2A, 2B)                (11A, 1B)
#        /     \                  /        \
# (y < 11)   (y >= 11)     (y < 10.75)     (y >= 10.75)
# (2A)       (2B)          (1A, 1B)        (10A)
#                          /      \
#                 (x < 3.5)        (x >= 3.5)
#                      (1B)        (1A)

do_assert("best split 3", tuple(dt.root.best_split[:2]),
          ('y', 12.5))
do_assert("best split 4", tuple(dt.root.low.best_split[:2]),
          ('x', 2.5))
do_assert("low low best split", tuple(dt.root.low.low.best_split[:2]),
          ('y', 11.25))
do_assert("low high best split", tuple(dt.root.low.high.best_split[:2]),
          ('y', 10.75))
do_assert("low high low best split", tuple(dt.root.low.high.low.best_split[:2]),
          ('x', 3.5))

do_assert("classify 1", dt.classify({'x': 2, 'y': 11.5}),
          'B')
do_assert("classify 2", dt.classify({'x': 2.5, 'y': 13}),
          'B')
do_assert("classify 3", dt.classify({'x': 4, 'y': 12}),
          'A')
do_assert("classify 4", dt.classify({'x': 3.25, 'y': 10.5}),
          'B')
do_assert("classify 5", dt.classify({'x': 3.75, 'y': 10.5}),
          'A')

print(cstring("&6All initial testspassed!"))

def test_dataset(df):
    dt_random = DecisionTree("random", dependent_variable="label")
    dt_random.fit(df)

    accurate = 0
    for row in df.to_json():
        l = dt_random.classify(row)
        if l == row['label']:
            accurate += 1
    return accurate/ df.get_length()

dataset = [(x, y, 'positive' if x*y>0 else 'negative') for x in range(-5, 6) for y in range(-5, 6) if 0 not in (x, y)]
df = DataFrame.from_array(dataset, ['x', 'y', 'label'])
do_assert("rf 1", test_dataset(df), 1)

dataset = [(x, y, 'A') for x in range(-5, 6) for y in range(-5, 6) if 0 not in (x, y)]
dataset += [(x, y, 'B') for x in range(1, 6) for y in range(1, 6)]
dataset += [(x, y, 'B') for x in range(1, 6) for y in range(1, 6)]
df = DataFrame.from_array(dataset, ['x', 'y', 'label'])
do_assert("rf 2", round(test_dataset(df), 2), 0.83)

dataset = [(x, y, z, 'positive' if x*y*z>0 else 'negative') for x in range(-5, 6) for y in range(-5, 6) for z in range(-5, 6) if 0 not in (x, y, z)]
df = DataFrame.from_array(dataset, ['x', 'y', 'z', 'label'])
do_assert("rf 3", test_dataset(df), 1)

dataset = [(x, y, z, 'A') for x in range(-5, 6) for y in range(-5, 6) for z in range(-5, 6) if 0 not in (x, y, z)]
dataset += [(x, y, z, 'B') for x in range(1, 6) for y in range(1, 6) for z in range(1, 6)]
dataset += [(x, y, z, 'B') for x in range(1, 6) for y in range(1, 6) for z in range(1, 6)]
df = DataFrame.from_array(dataset, ['x', 'y', 'z', 'label'])
do_assert("rf 4", test_dataset(df), 0.9)

print(cstring("&6All random tests passed!"))
