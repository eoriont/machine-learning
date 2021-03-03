import sys
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
        grad *= sum(self.get_act_func(j)(k) for j, k in enumerate(i))
        return {edge: grad * i[j] for j, edge in enumerate(self.weights)}

    def update_weights(self, data_point, learning_rate=0.01):
        grad = self.calc_gradient(data_point)
        self.weights = {edge: w - learning_rate * grad[edge] for edge, w in self.weights.items()}
        return self.weights

    def get_act_func(self, node):
        return self.activation_functions[self.activation_types[node]]['function']

    def get_act_deriv(self, node):
        return self.activation_functions[self.activation_types[node]]['derivative']


if __name__ == "__main__":
    # weights = {(0,2): -0.1, (1,2): 0.5}
    # nn = neuralnetwork(weights)
    # data_points = [
    #     {'input': [1,0], 'output': [1]},
    #     {'input': [1,1], 'output': [3]},
    #     {'input': [1,2], 'output': [5]},
    #     {'input': [1,3], 'output': [7]}
    # ]
    # for _ in range(1000):
    #     for data_point in data_points:
    #         nn.update_weights(data_point)

    # print(nn.weights)
    # # should be really close to
    # #     {(0,2): 1, (1,2): 2}

    import math
    weights = {(0,2): -0.1, (1,2): 0.5}

    def linear_function(x):
        return x
    def linear_derivative(x):
        return 1
    def sigmoidal_function(x):
        return 1/(1+math.exp(-x))
    def sigmoidal_derivative(x):
        s = sigmoidal_function(x)
        return s * (1 - s)

    activation_types = ['linear', 'linear', 'sigmoidal']
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

    nn = NeuralNetwork(weights, activation_types, activation_functions)

    data_points = [
    {'input': [1,0], 'output': [0.1]},
    {'input': [1,1], 'output': [0.2]},
    {'input': [1,2], 'output': [0.4]},
    {'input': [1,3], 'output': [0.7]}
    ]
    for i in range(1,10001):
        err = 0
        for data_point in data_points:
            nn.update_weights(data_point)
            err += nn.calc_squared_error(data_point)
        if i < 5 or i % 1000 == 0:
            print('iteration {}'.format(i))
            print('    gradient: {}'.format(nn.calc_gradient(data_point)))
            print('    updated weights: {}'.format(nn.weights))
            print('    error: {}'.format(err))
            print()

    print(nn.weights)
    # should be close to
    #     {(0,2): -2.44, (1,2): 1.07}

    # because the data points all lie approximately on the sigmoid
    #     output = 1/(1 + e^(-(input[0] * -2.44 + input[1] * 1.07)) )
