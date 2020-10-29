import sys
sys.path.append('src')
try:
    from dataframe import DataFrame
    from k_nearest_neighbors_classifier import KNearestNeighborsClassifier
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
except ImportError as e:
    print(e)


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
names = ['Cookie Type', 'Portion Eggs',
         'Portion Butter', 'Portion Sugar', 'Portion Flour']
df = DataFrame.from_array(data, names)
percents = []
for i in range(1, df.get_length()):
    trials = []
    for j in range(df.get_length()):
        df1, entry = df.remove_entry(j)
        knn = KNearestNeighborsClassifier(df1, prediction_column="Cookie Type")
        trials.append(knn.classify(entry, k=i) == entry["Cookie Type"])
    percents.append(trials)

percents = [t.count(True)/len(t) for t in percents]

plt.plot([i for i in range(len(percents))], percents, zorder=1)
plt.show()
