import sys
sys.path.append('src')
try:
    from matrix import Matrix
except ImportError as err:
    print(err)


def do_assert(test_name, output, expected):
    assert output == expected, f"Test {test_name} failed: output {output} expected to be {expected}"
    print(f"Test {test_name} PASSED!")


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

result = Matrix([[1, 0], [0, 1]])
do_assert("RREF 2x2", m1.rref(), result)

result = -8
do_assert("determinant 2x2", m1.det(), result)
do_assert("recursive determinant 2x2", m1.recursive_determinant(), result)

result = Matrix([[-3/4, 1/2], [5/8, -1/4]])
do_assert("inverse 2x2", m1.inverse(), result)
do_assert("inverse by minors 2x2", m1.inverse_by_minors(), result)

print("All tests passed!")
