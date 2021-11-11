# basic linear algebra calculator with numpy

import numpy as np
from numpy.matrixlib.defmatrix import matrix

# main function

def main():
    print("1. Vector Addition")
    print("2. Vector Subtraction")
    print("3. Vector Multiplication")
    print("4. Vector Norm")
    print("5. Vector Dot Product")
    print("6. Vector Cross Product")
    print("7. Matrix Addition")
    print("8. Matrix Subtraction")
    print("9. Matrix Multiplication")
    print("10. Matrix Transpose")
    print("11. Matrix Inverse")
    print("12. Matrix Determinant")
    print("13. Matrix Eigenvalues")
    print("14. Matrix Eigenvectors")
    print("15. Matrix Rank")
    print("16. Matrix Condition Number")
    print("17. Matrix Singular Value Decomposition")
    print("18. Matrix Nullspace")
    print("19. Matrix Pseudoinverse")
    print("20. Matrix QR Decomposition")
    print("21. Gaussian Elimination")

    if input("Enter the number of the function you want to use: ") == "1":
        vector_addition()
    elif input("Enter the number of the function you want to use: ") == "2":
        vector_subtraction()
    elif input("Enter the number of the function you want to use: ") == "3":
        vector_multiplication()
    elif input("Enter the number of the function you want to use: ") == "4":
        vector_norm()
    elif input("Enter the number of the function you want to use: ") == "5":
        vector_dot_product()
    elif input("Enter the number of the function you want to use: ") == "6":
        vector_cross_product()
    elif input("Enter the number of the function you want to use: ") == "7":
        matrix_addition()
    elif input("Enter the number of the function you want to use: ") == "8":
        matrix_subtraction()
    elif input("Enter the number of the function you want to use: ") == "9":
        matrix_multiplication()
    elif input("Enter the number of the function you want to use: ") == "10":
        matrix_transpose()
    elif input("Enter the number of the function you want to use: ") == "11":
        matrix_inverse()
    elif input("Enter the number of the function you want to use: ") == "12":
        matrix_determinant()
    elif input("Enter the number of the function you want to use: ") == "13":
        matrix_eigenvalues()
    elif input("Enter the number of the function you want to use: ") == "14":
        matrix_eigenvectors()
    elif input("Enter the number of the function you want to use: ") == "15":
        matrix_rank()
    elif input("Enter the number of the function you want to use: ") == "16":
        matrix_condition_number()
    elif input("Enter the number of the function you want to use: ") == "17":
        matrix_singular_value_decomposition()
    elif input("Enter the number of the function you want to use: ") == "18":
        matrix_nullspace()
    elif input("Enter the number of the function you want to use: ") == "19":
        matrix_pseudoinverse()
    elif input("Enter the number of the function you want to use: ") == "20":
        matrix_qr_decomposition()
    elif input("Enter the number of the function you want to use: ") == "21":
        gaussian_elimination()
    else:
        print("Invalid input")

def vector_input():
    print("Enter the vector in the form [x1, x2, ..., xn]")
    vector = np.array(input("Enter the vector: "))
    return vector


def matrix_input():
    n = int(input("Enter the number of rows: "))
    m = int(input("Enter the number of columns: "))
    matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            print("Enter the element at row", i+1, "and column", j+1, ": ")
            matrix[i][j] = float(input())
    return matrix

def vector_addition():
    print("How many Vectors?")
    n = int(input())
    vectors = []
    for i in range(n):
        vectors.append(vector_input())
    sum = np.zeros(len(vectors[0]))
    for i in range(len(vectors)):
        sum = np.add(sum, vectors[i])
    print("The sum of the vectors is: ", sum)

def vector_subtraction():
    print("How many Vectors?")
    n = int(input())
    vectors = []
    for i in range(n):
        vectors.append(vector_input())
    difference = np.zeros(len(vectors[0]))
    for i in range(len(vectors)):
        difference = np.subtract(vectors[i], difference)
    print("The difference of the vectors is: ", difference)

def vector_multiplication():
    print("Vector?")
    vector = vector_input()
    print("Scalar?")
    scalar = float(input())
    print("The product of the vector and scalar is:", np.multiply(vector, scalar))

def vector_norm():
    print("Vector?")
    vector = vector_input()
    print("The norm of the vector is:", np.linalg.norm(vector))

def vector_dot_product():
    print("How many Vectors?")
    n = int(input())
    vectors = []
    for i in range(n):
        vectors.append(vector_input())
    dot_product = np.zeros(len(vectors[0]))
    for i in range(len(vectors)):
        dot_product = np.add(dot_product, vectors[i])
    print("The dot product of the vectors is: ", dot_product)

def vector_cross_product():
    print("How many Vectors?")
    n = int(input())
    vectors = []
    for i in range(n):
        vectors.append(vector_input())
    cross_product = np.zeros(len(vectors[0]))
    for i in range(len(vectors)):
        cross_product = np.cross(vectors[i], cross_product)
    print("The cross product of the vectors is: ", cross_product)

def matrix_addition():
    print("How many Matrices?")
    n = int(input())
    matrices = []
    for i in range(n):
        matrices.append(matrix_input())
    sum = np.zeros(len(matrices[0]))
    for i in range(len(matrices)):
        sum = np.add(sum, matrices[i]) 
    print("The sum of the matrices is: ", sum)

def matrix_subtraction():
    print("How many Matrices?")
    n = int(input())
    matrices = []
    for i in range(n):
        matrices.append(matrix_input())
    difference = np.zeros(len(matrices[0]))
    for i in range(len(matrices)):
        difference = np.subtract(matrices[i], difference)
    print("The difference of the matrices is: ", difference)

def matrix_multiplication():
    print("How many Matrices?")
    n = int(input())
    matrices = []
    for i in range(n):
        matrices.append(matrix_input())
    product = np.zeros(len(matrices[0]))
    for i in range(len(matrices)):
        product = np.matmul(matrices[i], product)
    print("The product of the matrices is: ", product)

def matrix_transpose():
    print("Matrix?")
    matrix = matrix_input()
    print("The transpose of the matrix is: ", np.transpose(matrix))

def matrix_inverse():
    print("Matrix?")
    matrix = matrix_input()
    print("The inverse of the matrix is: ", np.linalg.inv(matrix))

def matrix_determinant():
    print("Matrix?")
    matrix = matrix_input()
    print("The determinant of the matrix is: ", np.linalg.det(matrix))

def matrix_eigenvalues():
    print("Matrix?")
    matrix = matrix_input()
    print("The eigenvalues of the matrix are: ", np.linalg.eigvals(matrix))

def matrix_eigenvectors():
    print("Matrix?")
    matrix = matrix_input()
    print("The eigenvectors of the matrix are: ", np.linalg.eig(matrix))

def matrix_rank():
    print("Matrix?")
    matrix = matrix_input()
    print("The rank of the matrix is: ", np.linalg.matrix_rank(matrix))

def matrix_condition_number():
    print("Matrix?")
    matrix = matrix_input()
    print("The condition number of the matrix is: ", np.linalg.cond(matrix))

def matrix_singular_value_decomposition():
    print("Matrix?")
    matrix = matrix_input()
    print("The singular values of the matrix are: ", np.linalg.svd(matrix))

def matrix_nullspace():
    print("Matrix?")
    matrix = matrix_input()
    print("The nullspace of the matrix is: ", np.linalg.null(matrix))

def matrix_pseudoinverse():
    print("Matrix?")
    matrix = matrix_input()
    print("The pseudoinverse of the matrix is: ", np.linalg.pinv(matrix))

def matrix_qr_decomposition():
    print("Matrix?")
    matrix = matrix_input()
    print("The QR decomposition of the matrix is: ", np.linalg.qr(matrix))

def gaussian_elimination():
    print("Matrix?")
    matrix = matrix_input()
    print("The solution of the matrix is: ", np.linalg.solve(matrix))
