class Matrix:
    # region
    # Stored as array of arrays
    def __init__(self, elements=None, shape=None, fill=0):
        self.elements = Matrix.fill_matrix(
            shape, fill) if elements is None else elements
        self.shape = self.get_shape() if shape is None else shape

    # Take number, list of numbers, or lists of lists and return list of lists
    @staticmethod
    def format_matrix_array(arr):
        if isinstance(arr, Matrix):
            return arr
        elif isinstance(arr, list):
            if len(arr) == 0:
                return [arr]
            if isinstance(arr[0], list):
                return arr
            else:
                return [arr]
        else:
            return [[arr]]

    # Go from [1] to 1 or [[1, 2, 3]] to [1, 2, 3], etc
    @staticmethod
    def compress_matrix_array(arr):
        if not isinstance(arr, list) or len(arr) == 0:
            return arr
        elif len(arr) == 1:
            return Matrix.compress_matrix_array(arr[0])
        elif not isinstance(arr[0], list):
            return arr
        elif len(arr[0]) == 1:
            return Matrix.compress_matrix_array(
                Matrix(arr).transpose().elements)
        else:
            return arr

    # Fill matrix with a number or make identity
    @staticmethod
    def fill_matrix(shape, fill):
        new_fill = 0 if fill == "diag" else fill
        new_mat = [[new_fill for _ in range(shape[1])]
                   for _ in range(shape[0])]
        if fill == "diag":
            for i, row in enumerate(new_mat):
                row[i] = 1
        return new_mat

    # Display Matrix
    def show(self, label="Matrix"):
        print(self.__str__(label))

    def is_square(self):
        return self.shape[0] == self.shape[1]

    # Add two matrices of the same shape
    def add(self, mat2):
        if self.shape != mat2.shape:
            raise Exception(
                "Error: Cannot add 2 matrices with different shapes!")
        new_mat = []
        for i, row in enumerate(self.elements):
            new_mat.append([val + mat2.elements[i][j]
                            for j, val in enumerate(row)])
        return Matrix(new_mat)

    # Multiply all numbers in the matrix by a scalar
    def scale(self, scalar):
        new_mat = []
        for row in self.elements:
            new_mat.append([val * scalar for val in row])
        return Matrix(new_mat)

    # Add the negative matrix to self
    def subtract(self, mat2):
        return self.add(mat2.scale(-1))

    # Return shape of matrix and check if invalid
    def get_shape(self):
        rows = len(self.elements)
        if rows == 0:
            return (0, 0)
        cols = len(self.elements[0])
        for row in self.elements:
            if cols != len(row):
                raise Exception("Error: Invalid matrix!")
        return (rows, cols)

    # Multiply this matrix by mat2
    # Reverse the shapes because I wrote it the wrong way
    def multiply(self, mat2):
        dim1 = self.shape
        dim2 = mat2.shape
        if (dim1[1] != dim2[0]):
            raise Exception("Error: Cannot multiply these matrices!")
        new_dim = (dim1[0], dim2[1])
        new_mat = []
        for x in range(new_dim[0]):
            new_row = []
            for y in range(new_dim[1]):
                val = [self.elements[x][i] * mat2.elements[i][y]
                       for i in range(dim1[1])]
                new_row.append(sum(val))
            new_mat.append(new_row)
        return Matrix(new_mat)

    # Raise matrix to the num power
    def exponent(self, num):
        new_mat = self.copy()
        for _ in range(num-1):
            new_mat @= self
        return new_mat

    # Returns a copy of self to avoid mutation
    def copy(self):
        return Matrix(elements=[[num for num in row] for row in self.elements])

    # Get the smallest value in matrix
    def min(self):
        return self.filter_matrix(lambda out, val: out > val)

    # Get the largest value in matrix
    def max(self):
        return self.filter_matrix(lambda out, val: out < val)

    # Go through each element and get value based on lambda
    def filter_matrix(self, test_lambda):
        output = self.elements[0][0]
        for row in self.elements:
            for val in row:
                if test_lambda(output, val):
                    output = val
        return output

    # Reflect self over its main diagonal
    def transpose(self):
        return Matrix([list(arr) for arr in zip(*self.elements)])

    # Check if the lists of elements in each matrix are the same
    def is_equal(self, mat2):
        return self.elements == mat2.elements

    # Get slice of matrix
    def get(self, pos, compress=True):
        rows, cols = pos

        # Ensure list takes in list of lists or
        # just list and gives out list of lists
        def ensure_list(l, i): return l[i] if isinstance(i, slice) else [l[i]]

        yrange = ensure_list(self.elements, rows)
        new_mat = [ensure_list(row, cols) for row in yrange]
        return Matrix.compress_matrix_array(new_mat) if compress else new_mat

    # Set certain matrix pattern to other matrix
    def set(self, pos, val):
        val = Matrix(Matrix.format_matrix_array(val))
        shape = Matrix(self.get(pos, False)).shape
        if 0 in shape and 0 in val.shape:
            return
        # Transpose if array is in wrong direction
        if 1 in shape and 1 in val.shape:
            if not shape.index(1) == val.shape.index(1):
                val = val.transpose()
        # Can't set matrix to wrong shape
        if val.shape != shape:
            raise Exception("Cannot set value to matrix of non equal shape!")

        # Ensure range takes in a slice or int
        # (i for index) and gives out a range
        def ensure_range(i, end): return range(
            end+1)[i] if isinstance(i, slice) else range(i, i+1)

        row_range = list(
            zip(ensure_range(pos[0], self.shape[0]), range(shape[0])))
        col_range = list(
            zip(ensure_range(pos[1], self.shape[1]), range(shape[1])))
        for row, row_idx in row_range:
            for x, x_idx in col_range:
                self.elements[row][x] = val.elements[row_idx][x_idx]

    # Swap row1 with row2
    def swap_rows(self, row1, row2):
        temp_row = self[row1, :]
        self[row1, :] = self[row2, :]
        self[row2, :] = temp_row

    # Divides a row by its first non zero entry
    def scale_row(self, row_idx):
        row = self[row_idx, :]
        if type(row) != list:
            row = [row]
        _, divisor = Matrix.get_first_non_zero_entry(row)
        row = [i/divisor for i in row]
        return row, divisor

    # Gets the first row with non zero item in col,
    # and to the left are all zeros
    def get_pivot_row(self, col):
        for i, row in enumerate(self.elements):
            j, _ = Matrix.get_first_non_zero_entry(row)
            if col == j:
                return i
        return None

    # Subtracts multiples of clear_row from rows below it
    def clear_below(self, clear_row_idx):
        return self.clear_rows(
            clear_row_idx, self.elements[clear_row_idx+1:], clear_row_idx+1)

    # Subtracts multiples of clear_row from rows above it
    def clear_above(self, clear_row_idx):
        return self.clear_rows(clear_row_idx, self.elements[:clear_row_idx], 0)

    # This is used by clear_above and clear_below for minimal duplicated code
    def clear_rows(self, clear_row_idx, rows_to_clear, clear_row_offset):
        col_idx, divisor = Matrix.get_first_non_zero_entry(
            self.elements[clear_row_idx])
        # If pivot is gone, do nothing
        if None in [col_idx, divisor]:
            return self
        new_mat = self.copy()
        clear_row = new_mat.elements[clear_row_idx]

        def possibly_int(x): return int(x) if x.is_integer() else x

        # Loop through all rows in rows_to_clear the clear
        # row and set row in self to the cleared row
        for row_idx, row in enumerate(rows_to_clear):
            multiple = row[col_idx]/divisor
            new_row = [possibly_int(x-(multiple*clear_row[i]))
                       for i, x in enumerate(row)]
            new_mat.elements[row_idx+clear_row_offset] = new_row
        return new_mat

    # Gets the first non zero entry in a list and returns (index, item)
    @ staticmethod
    def get_first_non_zero_entry(arr):
        for i, x in enumerate(arr):
            if x != 0:
                return i, x
        return None, None

    # Follow an algorithm to make matrix into reduced row echelon form
    # If return_determinant is true, this function will return the
    # determinant of self
    def rref(self, return_determinant=False):
        if not self.is_square() and return_determinant:
            raise Exception(
                "Error: Can't find the shape of a non square matrix!")
        new_mat = self.copy()
        divisors_product = 1
        for i in range(new_mat.shape[0]):
            pivot_row = new_mat.get_pivot_row(i)
            if pivot_row is None:
                if return_determinant:
                    return 0
                continue
            if pivot_row != i:
                divisors_product *= -1
                new_mat.swap_rows(pivot_row, i)
            row, divisor = new_mat.scale_row(i)
            new_mat[i, :] = row
            divisors_product *= divisor
            new_mat = new_mat.clear_below(i)
        for j in range(new_mat.shape[0]-1, 0, -1):
            new_mat = new_mat.clear_above(j)
        if return_determinant:
            return divisors_product
        return new_mat

    # Augments self with an identity matrix, turns into reduced row echelon
    # form, then returns the augmented part
    def inverse(self):
        if not self.is_square():
            raise Exception(
                "Error: Matrix not invertible because it's not a square!")
        if self.det() == 0:
            raise Exception(
                "Error: Cannot find inverse of a matrix whose determinant is 0!")
        new_mat = Matrix(shape=(self.shape[0], self.shape[1]*2), fill=0)
        size = new_mat.shape[0]
        new_mat[:, :size] = self.elements
        new_mat[:, size:] = Matrix(shape=(size, size), fill="diag").elements
        new_mat = new_mat.rref()
        return Matrix(Matrix.format_matrix_array(new_mat[:, size:]))

    # Solve Ax = b
    def solve(self, column_vector):
        inverse = self.inverse()
        col_mat = Matrix(Matrix.format_matrix_array(column_vector)).transpose()
        return Matrix.compress_matrix_array((inverse @ col_mat).elements)

    # Do an operation on every value in the matrix and return a new matrix
    def element_operation(self, op):
        new_mat = self.copy()
        for i, row in enumerate(new_mat.elements):
            for j, _ in enumerate(row):
                new_mat.elements[i][j] = op(new_mat.elements[i][j], (i, j))
        return new_mat
    # endregion

    def recursive_determinant(self):
        if self.shape[0] != self.shape[1]:
            raise Exception(
                "Error: Can't find the shape of a non square matrix!")
        if self.shape == (2, 2):
            return self[0, 0]*self[1, 1] - self[1, 0]*self[0, 1]
        if self.shape == (1, 1):
            return self[0, 0]
        determinant = 0
        for i, element in enumerate(self[0, :]):
            A = Matrix(shape=(self.shape[0]-1, self.shape[0]-1))
            A[:, :i] = self[1:, :i]
            A[:, i:] = self[1:, i+1:]
            small_determinant = A.recursive_determinant()
            determinant += (-1)**(i) * element * small_determinant
        return determinant

    # Shortcut to the rref determinant
    def det(self):
        return self.rref(True)

    def inverse_by_minors(self):
        self_det = self.det()
        if self_det == 0:
            raise Exception(
                "Error: Cannot find inverse of a matrix whose determinant is 0!")
        minors_dets_mat = Matrix(shape=self.shape, fill=0)
        for i, row in enumerate(self.elements):
            for j, _ in enumerate(row):
                minors_dets_mat.elements[i][j] = (-1)**(i + j) * \
                    self.get_minor((i, j)).det()
        adjugate = minors_dets_mat.transpose()
        return adjugate * (1/self_det)

    # Get minor of self (get submatrix minus row and column in pos)
    def get_minor(self, pos):
        x, y = pos
        A = Matrix(shape=(self.shape[0]-1, self.shape[0]-1))
        A[:x, :y] = self[:x, :y]
        A[:x, y:] = self[:x, y+1:]
        A[x:, y:] = self[x+1:, y+1:]
        A[x:, :y] = self[x+1:, :y]
        return A

    # region
    # Overload the getitem op

    def __getitem__(self, pos):
        return self.get(pos)

    # Overload the setitem op
    def __setitem__(self, pos, val):
        self.set(pos, val)

    # Overload the equals op
    def __eq__(self, mat2):
        return self.is_equal(mat2)

    # Overload the add op
    def __add__(self, mat2):
        return self.add(mat2)

    # Overload the subtract op
    def __sub__(self, mat2):
        return self.subtract(mat2)

    # Overload the matrix multiply op
    def __matmul__(self, mat2):
        return self.multiply(mat2)

    # Overload the multiplication op for SCALING
    def __mul__(self, scalar):
        if (type(scalar) != int and type(scalar) != float):
            raise Exception("Error: Cannot scale by non-integer value")
        return self.scale(scalar)

    # Overload the pow op
    def __pow__(self, num):
        return self.exponent(num)

    # Overload the string op
    def __str__(self, label="Matrix"):
        string = f"{label} \n"
        for row in self.elements:
            string += str(row) + "\n"
        string += "________________"
        return string
    # endregion
