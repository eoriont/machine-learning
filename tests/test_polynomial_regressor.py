from otest import do_assert
import sys
sys.path.append('src')
try:
    from polynomial_regressor import PolynomialRegressor
except ImportError as err:
    raise err


def round_down(t, precision=6):
    if type(t) == tuple:
        return tuple(round(x, precision) for x in t)
    elif type(t) in (float, int):
        return round(t, precision)


data = [(0, 1), (1, 2), (2, 5), (3, 10), (4, 20), (5, 30)]

tests = [{'deg': 0, 'coefs': [11.333333333333332], 'eval': 11.333333333333332},
         {'deg': 1, 'coefs': [-3.2380952380952412,
                              5.828571428571428], 'eval': 8.419047619047616},
         {'deg': 2, 'coefs': [1.107142857142763, -0.6892857142856474,
                              1.3035714285714226], 'eval': 4.942857142857159},
         {'deg': 3, 'coefs': [1.1349206349217873, -0.8161375661377197,
                              1.3730158730155861, -0.009259259259233155], 'eval': 4.920634920634827},
         {'deg': 5, 'coefs': [0.9999999917480108, -2.950000002085698, 6.9583333345161265, -3.9583333337779045, 1.0416666667658463, -0.09166666667401097], 'eval': 4.999999990103076}]


for test in tests:
    deg = test['deg']
    regressor = PolynomialRegressor(degree=deg)
    regressor.ingest_data(data)
    regressor.solve_coefficients()
    test_name = f"solve_coefficients degree = {deg}"
    do_assert(test_name, round_down(regressor.coefficients),
              round_down(test['coefs']))
    test_name = f"evaluate degree = {deg}"
    do_assert(test_name, round_down(regressor.evaluate(2)),
              round_down(test['eval']))

print("All tests passed!")
