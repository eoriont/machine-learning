from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = [[0.14, 0.14, 0.28, 0.44],
        [0.22, 0.1, 0.45, 0.33],
        [0.1, 0.19, 0.25, 0.4],
        [0.02, 0.08, 0.43, 0.45],
        [0.16, 0.08, 0.35, 0.3],
        [0.14, 0.17, 0.31, 0.38],
        [0.05, 0.14, 0.35, 0.5],
        [0.1, 0.21, 0.28, 0.44],
        [0.04, 0.08, 0.35, 0.47],
        [0.11, 0.13, 0.28, 0.45],
        [0.0, 0.07, 0.34, 0.65],
        [0.2, 0.05, 0.4, 0.37],
        [0.12, 0.15, 0.33, 0.45],
        [0.25, 0.1, 0.3, 0.35],
        [0.0, 0.1, 0.4, 0.5],
        [0.15, 0.2, 0.3, 0.37],
        [0.0, 0.13, 0.4, 0.49],
        [0.22, 0.07, 0.4, 0.38],
        [0.2, 0.18, 0.3, 0.4]]

sum_squared_errors = []
for k_value in range(1, 6):
    # Populate initial clusters
    initial_clusters = {i: [] for i in range(k_value)}
    for i in range(len(data)):
        initial_clusters[i%k_value].append(i)

    # Fit kmeans model
    kmeans = KMeans(n_clusters=k_value)
    kmeans.fit(data)

    sum_squared_errors.append(kmeans.inertia_)

plt.xlabel("k")
plt.ylabel("sum squared error")
plt.plot(list(range(1, 6)), sum_squared_errors)
plt.savefig("a128")
