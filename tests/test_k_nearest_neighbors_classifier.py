import sys
sys.path.append('src/models')
from dataframe import DataFrame
from k_nearest_neighbors_classifier import KNearestNeighborsClassifier
from otest import do_assert

df = DataFrame.from_array(
    [['Shortbread',     0.14,       0.14,      0.28,     0.44],
     ['Shortbread',     0.10,       0.18,      0.28,     0.44],
     ['Shortbread',     0.12,       0.10,      0.33,     0.45],
     ['Shortbread',     0.10,       0.25,      0.25,     0.40],
     ['Sugar',     0.00,       0.10,      0.40,     0.50],
     ['Sugar',     0.00,       0.20,      0.40,     0.40],
     ['Sugar',     0.10,       0.08,      0.35,     0.47],
     ['Sugar',     0.00,       0.05,      0.30,     0.65],
     ['Fortune',     0.20,       0.00,      0.40,     0.40],
     ['Fortune',     0.25,       0.10,      0.30,     0.35],
     ['Fortune',     0.22,       0.15,      0.50,     0.13],
     ['Fortune',     0.15,       0.20,      0.35,     0.30],
     ['Fortune',     0.22,       0.00,      0.40,     0.38]],
    ['Cookie Type', 'Portion Eggs',
     'Portion Butter', 'Portion Sugar', 'Portion Flour']
)

knn = KNearestNeighborsClassifier(k=5)
knn.fit(df, dependent_variable='Cookie Type')
observation = {
    'Portion Eggs': 0.10,
    'Portion Butter': 0.15,
    'Portion Sugar': 0.30,
    'Portion Flour': 0.45
}

# Returns a dataframe representation of the following array:
do_assert("distances", knn.compute_distances(observation).apply(
    'distances', lambda x: round(x, 3)).to_array(),
    [[0.047, 'Shortbread'],
     [0.037, 'Shortbread'],
     [0.062, 'Shortbread'],
     [0.122, 'Shortbread'],
     [0.158, 'Sugar'],
     [0.158, 'Sugar'],
     [0.088, 'Sugar'],
     [0.245, 'Sugar'],
     [0.212, 'Fortune'],
     [0.187, 'Fortune'],
     [0.396, 'Fortune'],
     [0.173, 'Fortune'],
     [0.228, 'Fortune']])

# Note: the above has been rounded to 3 decimal places for ease of viewing, but you should not round yourself.

# Returns a dataframe representation of the following array:
do_assert("ordered distances", knn.nearest_neighbors(observation).apply('distances', lambda x: round(x, 3)).to_array(),
          [[0.037, 'Shortbread'],
           [0.047, 'Shortbread'],
           [0.062, 'Shortbread'],
           [0.088, 'Sugar'],
           [0.122, 'Shortbread'],
           [0.158, 'Sugar'],
           [0.158, 'Sugar'],
           [0.173, 'Fortune'],
           [0.187, 'Fortune'],
           [0.212, 'Fortune'],
           [0.228, 'Fortune'],
           [0.245, 'Sugar'],
           [0.396, 'Fortune']])

# Note: the above has been rounded to 3 decimal places for ease of viewing, but you should not round yourself.

do_assert("average distances", {k: round(v, 3) for k, v in knn.compute_average_distances(observation).items()},
          {
    'Shortbread': 0.067,
    'Sugar': 0.162,
    'Fortune': 0.239
})

# Note: the above has been rounded to 3 decimal places for ease of viewing, but you should not round yourself.

do_assert("classify", knn.classify(observation),
          'Shortbread')

# (In the case of a tie, chose whichever class has a lower average distance. If that is still a tie, then pick randomly.)
