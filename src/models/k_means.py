class KMeans:
    def __init__(self, initial_clusters, data):
        self.clusters = initial_clusters
        self.last_clusters = initial_clusters
        self.data = data
        self.update_cluster_centers()

    def run(self, max_iters=100):
        for _ in range(max_iters):
            self.update_clusters_once()

            identical = True
            for k in self.clusters:
                if set(self.clusters[k]) != set(self.last_clusters[k]):
                    identical = False
                    break
            if identical:
                break

    def update_clusters_once(self):
        new_clusters = {cluster_id: [] for cluster_id in self.clusters}
        for row_id, row in enumerate(self.data):
            closest_cluster = self.get_closest_cluster(row)
            new_clusters[closest_cluster].append(row_id)
        self.last_clusters = self.clusters
        self.clusters = new_clusters
        self.update_cluster_centers()

    def get_closest_cluster(self, row):
        distances = self.get_cluster_distances(row)
        return min(distances, key=distances.get)

    def get_cluster_distances(self, row):
        return {
            cluster_id: (sum((val-center)**2 for val, center in zip(row, cluster_center)))
            for cluster_id, cluster_center in self.centers.items()
        }


    def update_cluster_centers(self):
        self.centers = {
            cluster_index: [
                sum(self.data[row][col] for row in cluster)/len(cluster)
                for col in range(len(self.data[0]))
            ]
            for cluster_index, cluster in self.clusters.items()
        }

    def get_cluster_squared_errors(self):
        self.update_cluster_centers()
        return sum(
            self.get_cluster_distances(self.data[row])[cluster_id]
            for cluster_id, cluster in self.clusters.items()
            for row in cluster
        )
