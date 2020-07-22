from otest import do_assert, assert_exception
import sys
sys.path.append('src')
try:
    from matrix import Matrix
except ImportError as err:
    print(err)


# ARITHMETIC TESTS
m1 = Matrix([[2, 4], [5, 6]])
m2 = Matrix([[1, 2], [3, 4]])
result = Matrix([[3, 6], [8, 10]])
do_assert("addition 2x2", m1 + m2, result)

result = Matrix([[1, 2], [2, 2]])
do_assert("subtraction 2x2", m1 - m2, result)

result = Matrix([[14, 20], [23, 34]])
do_assert("multiplication 2x2", m1 @ m2, result)

scalar = 10
result = Matrix([[20, 40], [50, 60]])
do_assert("scaling 2x2", m1 * scalar, result)

result = Matrix([[2, 5], [4, 6]])
do_assert("transpose 2x2", m1.transpose(), result)

# RREF TESTS
m3 = Matrix([[5, 0, 9], [8, 0, 1], [1, 2, 0]])
result = Matrix([[1, 0, 0], [0, 1, 0], [-0.0, -0.0, 1.0]])
do_assert("RREF 3x3", m3.rref(), result)

m3 = Matrix([[5, 4, 6, 3, 10], [5, 5, 1, 9, 9], [3, 3, 4, 10, 7]])
result = Matrix([[1, 0, 0, -11.235294117647065, 0.35294117647058765], [0, 1, 0, 12.764705882352946,
                                                                       1.3529411764705888], [0.0, 0.0, 1.0, 1.3529411764705892, 0.47058823529411775]])
do_assert("RREF 3x5", m3.rref(), result)

m3 = Matrix([[10, 10, 6], [9, 4, 7], [1, 10, 8], [1, 3, 5], [8, 2, 3]])
result = Matrix([[1, 0, 0], [0, 1, 0], [0.0, 0.0, 1.0], [0, 0, 0], [0, 0, 0]])
do_assert("RREF 5x3", m3.rref(), result)

m0 = Matrix([[7, 6, 5], [3, 0, 4], [1, 2, 7], [2, 1, 0], [8, 3, 6]])
result = Matrix([[1, 0, 0], [0, 1, 0], [0.0, 0.0, 1.0], [0, 0, 0], [0, 0, 0]])
do_assert("RREF 5x3", m0.rref(), result)

m0 = Matrix([[9, 1, 6, 6, 10], [1, 9, 9, 1, 7], [
            2, 4, 3, 9, 5], [1, 7, 3, 5, 0], [8, 3, 10, 7, 7]])
result = Matrix([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [
                0, 0, 0, 1, 0], [-0.0, -0.0, -0.0, -0.0, 1.0]])
do_assert("RREF 5x5", m0.rref(), result)

m0 = Matrix([[7, 10, 2, 9, 10], [4, 3, 6, 2, 5], [2, 9, 5, 9, 4]])
result = Matrix([[1, 0, 0, -0.24232081911262804, 1.1399317406143346], [0, 1, 0, 1.0784982935153584,
                                                                       0.2081911262798635], [0.0, 0.0, 1.0, -0.044368600682593795, -0.030716723549488064]])
do_assert("RREF 3x5", m0.rref(), result)


# Inverse / Determinant Tests

# Matrix with det = 0
m8 = Matrix([[1, 2, 3], [1, 2, 4], [0, 0, 0]])
result = 0
do_assert("determinant 3x3", m8.det(), result)
do_assert("recursive determinant 3x3", m8.recursive_determinant(), result)

assert_exception("inverse of dependent row 3x3", m8.inverse)
assert_exception("inverse by minors of dependent row 3x3",
                 m8.inverse_by_minors)

# Invertable Matrices
m8 = Matrix([[9, 3, 8], [5, 1, 6], [10, 0, 1]])
# result = 93.99999999999999
result = -8
do_assert("determinant 2x2", m1.det(), result)
do_assert("recursive determinant 2x2", m1.recursive_determinant(), result)

result = Matrix([[0.010638297872340413, -0.03191489361702121, 0.10638297872340424], [0.5851063829787235, -
                                                                                     0.7553191489361706, -0.14893617021276595], [-0.1063829787234043, 0.3191489361702128, -0.06382978723404255]])
do_assert("inverse 3x3", m8.inverse(), result)
do_assert("inverse by minors 3x3", m8.inverse_by_minors().element_operation(
    lambda x, _: round(x, 5)), result.element_operation(lambda x, _: round(x, 5)))

# Non square matrices
assert_exception("determinant 5x3", m3.det)
assert_exception("recursive determinant 5x3", m3.recursive_determinant)
assert_exception("inverse 5x3", m3.inverse)
assert_exception("inverse by minors 5x3", m3.inverse_by_minors)

m = Matrix([[1, 2, 3, 4],
            [5, 0, 6, 0],
            [0, 7, 0, 8],
            [9, 0, 0, 10]])
should_be_identity = (
    m@m.inverse()).element_operation(lambda x, _: round(x, 6))
do_assert("4x4 matrix * inverse == identity",
          should_be_identity, Matrix(shape=(4, 4), fill="diag"))

m = Matrix([[1.2, 5.3, 8.9, -10.3, -15],
            [3.14, 0, -6.28, 0, 2.71],
            [0, 1, 1, 2, 3],
            [5, 8, 13, 21, 34],
            [1, 0, 0.5, 0, 0.1]])
should_be_identity = (
    m@m.inverse()).element_operation(lambda x, _: round(x, 6))
do_assert("5x5 matrix * inverse == identity",
          should_be_identity, Matrix(shape=(5, 5), fill="diag"))

print("All tests passed!")
