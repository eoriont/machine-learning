import sys
sys.path.append("src/models")
from neural_network import NeuralNetwork
from otest import do_assert

def round_weights(weights, places=1):
    return {e: round(w, places) for e, w in weights.items()}

weights = {(0, 2): -0.1, (1, 2): 0.5}
nn = NeuralNetwork(weights)
do_assert("predict", nn.predict([1, 3]), 1.4)

data_point = {'input': [1, 3], 'output': [7]}
do_assert("calc_squared_error", round(nn.calc_squared_error(data_point), 2), 31.36)

do_assert("calc_gradient", round_weights(nn.calc_gradient(data_point)), {(0,2): -11.2, (1,2): -33.6})

do_assert("update_weights", round_weights(nn.update_weights(data_point, learning_rate=0.01), 3), {(0,2): 0.012, (1,2): 0.836})
