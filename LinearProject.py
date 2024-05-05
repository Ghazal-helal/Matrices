#Matrix Calculator
import tkinter as tk
#function to add 2 matrices together
def MatrixAddition(m1, m2):
    # Check if matrices have the same dimensions
    if len(m1) != len(m2) or len(m1[0]) != len(m2[0]):
        return "you have to enter Matrices of the same dimensions to add"

    # Initialize the result matrix as 0
    result = [[0] * len(m1[0]) for _ in range(len(m1))]

    # Add the corresponding elements of the matrices
    for x in range(len(m1)):
        for y in range(len(m1[0])):
            result[x][y] = m1[x][y] + m2[x][y]

    return result

#function to multiply 2 matrices together
def MatrixMultiplication(m1, m2):
    # Check if matrices can be computed
    if len(m1[0]) != len(m2):
        return "1st matrix columns should equal 2nd matrix rows"

    # Initialize the result matrix with zeros
    result = [[0] * len(matrix2[0]) for _ in range(len(matrix1))]

    # Perform matrix multiplication
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                result[i][j] += matrix1[i][k] * matrix2[k][j]

    return result

#function to transpose the matrix
def MatrixTranspose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

#function to find the determinant of the matrix for inverse
def determinant2x2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

#function to find the inverse of the matrix
def InverseMatrix(matrix):
    det = determinant(matrix)

    if det == 0:
        return "Matrix cannot be invered"

    adjugate = [
        [matrix[1][1], -matrix[0][1]],
        [-matrix[1][0], matrix[0][0]]
    ]

    inverse = [[adjugate[i][j] / det for j in range(2)] for i in range(2)]

    return inverse

#function to find the determinant of the matrix
def determinant(matrix):
    size = len(matrix)

    #If the matrix is 1x1, return its only element
    if size == 1:
        return matrix[0][0]

    #If the matrix is 2x2, return the determinant using the formula
    if size == 2:
        return determinant2x2(matrix)

    det = 0
    for col in range(size):
        det += ((-1) ** col) * matrix[0][col] * determinant([row[:col] + row[col + 1:] for row in matrix[1:]])

    return det

def matrix_inverse(matrix):
    size = len(matrix)

    if size != len(matrix[0]):
        return "Matrix is not square, and inverse is not defined"

    det = determinant(matrix)

    if det == 0:
        return "Matrix is not invertible"

    adjugate = []
    for i in range(size):
        row = []
        for j in range(size):
            cofactor = ((-1) ** (i + j)) * determinant([row[:j] + row[j + 1:] for row in (matrix[:i] + matrix[i + 1:])])
            row.append(cofactor)
        adjugate.append(row)

    adjugate_transpose = MatrixTranspose(adjugate)

    inverse = [[adjugate_transpose[i][j] / det for j in range(size)] for i in range(size)]

    return inverse

#function for user to enter the matrix rows and columns
def inputMatrix(rows, cols):
    matrix = []
    print(f"Enter {rows}x{cols} matrix elements:")
    for i in range(rows):
        row = []
        for j in range(cols):
            element = float(input(f"Enter element at position ({i+1}, {j+1}): "))
            row.append(element)
        matrix.append(row)
    return matrix


#RESULTS
# matrix size
rows_matrix1 = int(input("Enter the number of rows for 1st Matrix: "))
cols_matrix1 = int(input("Enter the number of columns for 1st Matrix: "))

rows_matrix2 = int(input("Enter the number of rows for 2nd Matrix: "))
cols_matrix2 = int(input("Enter the number of columns for 2nd Matrix: "))

# Input matrices from the user
matrix1 = inputMatrix(rows_matrix1, cols_matrix1)
matrix2 = inputMatrix(rows_matrix2, cols_matrix2)

# compute matrices operations
addition_result = MatrixAddition(matrix1, matrix2)
multiplication_result = MatrixMultiplication(matrix1, matrix2)
transpose_result = MatrixTranspose(matrix1)
inverse_result = matrix_inverse(matrix1)
determinant_result = determinant(matrix1)

# Display the results
print("\nMatrix1:\n", matrix1)
print("\nMatrix2:\n", matrix2)

print("\nAddition Result:\n", addition_result)
print("\nMultiplication Result:\n", multiplication_result)
print("\nTranspose Result:\n", transpose_result)
print("\nInverse Result:\n", inverse_result)
print("\nDeterminant Result:", determinant_result)



