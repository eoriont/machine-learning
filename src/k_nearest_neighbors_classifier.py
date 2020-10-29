from dataframe import DataFrame


class KNearestNeighborsClassifier:
    def __init__(self, df, prediction_column):
        self.df = df
        self.prediction_column = prediction_column

    def compute_distances(self, observation):
        rows = self.df.remove_columns([self.prediction_column]).to_array()
        types = self.df.get_column(self.prediction_column)
        c = [x for x in self.df.columns if x != self.prediction_column]
        obs = [observation[col] for col in c]
        return DataFrame({'distances': [dist(obs, rows[i]) for i in range(len(rows))], 'types': types}, ['distances', 'types'])

    def nearest_neighbors(self, observation):
        return self.compute_distances(observation).order_by('distances', ascending=True)

    def compute_average_distances(self, obs, k=None):
        k = len(self.df.get_column(self.prediction_column)) if k is None else k
        distances = self.compute_distances(obs).select_rows(range(k))
        types = distances.get_column('types')
        indices = {t: [i for i in range(
            len(types)) if types[i] == t] for t in set(types)}
        distances = distances.get_column('distances')
        avgs = {t: sum(distances[i] for i in indices[t])/len(indices[t])
                for t in set(types)}
        return avgs

    def classify(self, obs, k):
        near = self.nearest_neighbors(obs).select_rows(
            range(k)).get_column("types")
        type_counts = {t: near.count(t) for t in set(near)}
        m = max(type_counts, key=type_counts.get)
        if list(type_counts.values()).count(type_counts[m]) > 1:
            avgs = self.compute_average_distances(obs, k)
            return min(avgs, key=avgs.get)
        else:
            return m


def dist(p1, p2):
    return sum((x-y)**2 for x, y in zip(p1, p2))**0.5
