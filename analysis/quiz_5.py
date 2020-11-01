import sys
sys.path.append('src')
try:
    from dataframe import DataFrame
    from k_nearest_neighbors_classifier import KNearestNeighborsClassifier
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
except ImportError as e:
    print(e)

data = [['A', 1, 2],
        ['B', 3, 4],
        ['B', 4, 5],
        ['B', 5, 5],
        ['A', 1, 1]]
names = ['type', 'x', 'y']
df = DataFrame.from_array(data, names)
knn = KNearestNeighborsClassifier(df, prediction_column="type")
print(knn.classify({'x': 0, 'y': 0}, k=5))
