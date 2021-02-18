import sys
sys.path.append('src/graphs')
from weighted_graph import WeightedGraph

class NeuralNetwork(WeightedGraph):
    def __init__(self, weights):
        vertices = sorted(list(set(i for x in weights.keys() for i in x)))
        super().__init__(weights, vertices)

    def predict(self, inputs):
        # Gross hack
        #! Need a way to find the input layer and output layer
        return sum(self.weights[(i, 2)]*d for i, d in enumerate(inputs))

    def calc_squared_error(self, data_point):
        i, o = data_point['input'], data_point['output']
        return (o[0]-self.predict(i))**2

    def calc_gradient(self, data_point):
        i, o = data_point['input'], data_point['output']
        act_sub_pred = -2 * (o[0]-self.predict(i))
        return {edge: act_sub_pred * i[j] for j, edge in enumerate(self.weights.keys())}

    def update_weights(self, data_point, learning_rate=0.01):
        grad = self.calc_gradient(data_point)
        self.weights = {edge: w - learning_rate*grad[edge] for edge, w in self.weights.items()}
        return self.weights

if __name__ == "__main__":
    weights = {(0,2): -0.1, (1,2): 0.5}
    nn = NeuralNetwork(weights)
    data_points = [
        {'input': [1,0], 'output': [1]},
        {'input': [1,1], 'output': [3]},
        {'input': [1,2], 'output': [5]},
        {'input': [1,3], 'output': [7]}
    ]
    for _ in range(1000):
        for data_point in data_points:
            nn.update_weights(data_point)

    print(nn.weights)
    # should be really close to
    #     {(0,2): 1, (1,2): 2}
