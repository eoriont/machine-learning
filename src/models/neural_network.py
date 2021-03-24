import sys, math
sys.path.append('src/graphs')
from weighted_graph import WeightedGraph

class NeuralNetwork(WeightedGraph):
    def __init__(self, weights, activation_types, activation_functions):
        self.activation_types = activation_types
        self.activation_functions = activation_functions
        vertices = sorted(list(set(i for x in weights.keys() for i in x)))
        super().__init__(weights, vertices)

    def predict(self, inputs):
        #! Gross hack
        #! Need a way to find the input layer and output layer

        return self.get_act_func(2)(
            sum(self.weights[(i, 2)] * self.get_act_func(i)(d) for i, d in enumerate(inputs))
        )

    def calc_squared_error(self, data_point):
        i, o = data_point['input'], data_point['output']
        return (o[0] - self.predict(i))**2

    def calc_gradient(self, data_point):
        i, o = data_point['input'], data_point['output']
        grad = 2 * (self.predict(i) - o[0]) * self.get_act_deriv(2)(o[0])
        return {
            (0, 1): grad * self.weights[(1, 2)] *
        }
    def update_weights(self, data_point, learning_rate=0.01):
        grad = self.calc_gradient(data_point)
        self.weights = {edge: w - learning_rate * grad[edge] for edge, w in self.weights.items()}
        return self.weights

    def get_act_func(self, node):
        return self.activation_functions[self.activation_types[node]]['function']

    def get_act_deriv(self, node):
        return self.activation_functions[self.activation_types[node]]['derivative']

if __name__ == "__main__":
    def linear_function(x):
        return x
    def linear_derivative(x):
        return 1
    def sigmoidal_function(x):
        return 1/(1+math.exp(-x))
    def sigmoidal_derivative(x):
        s = sigmoidal_function(x)
        return s * (1 - s)
    activation_functions = {
        'linear': {
            'function': linear_function,
            'derivative': linear_derivative
        },
        'sigmoidal': {
            'function': sigmoidal_function,
            'derivative': sigmoidal_derivative
        }
    }
    weights = {(0, 1): 1, (1, 2): 1}
    nn = NeuralNetwork(weights, ['1', '1', '2'])
