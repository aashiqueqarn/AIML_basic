import numpy as np

def matrix_multiplication(A, B):
    rows_a, cols_a = A.shape
    rows_b, cols_b = B.shape
    assert rows_a == rows_b, "Shape mismatch"
    result = np.zeros((rows_a, cols_b))
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i, j] += A[i, k] * B[k, j]
    return result


def vector_operation():
    vector_a = np.array([2, 3])
    vector_b = np.array([-1, 4])
    print(vector_a + vector_b)
    print(np.linalg.norm(vector_a))
    print(np.linalg.norm(vector_b))

def cosine_similarity(a, b):
    normal_a = np.linalg.norm(a)
    normal_b = np.linalg.norm(b)
    normal = normal_a * normal_b
    return np.dot(a,b) / normal



if __name__ == '__main__':
    vector_operation()
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    print(matrix_multiplication(A, B))
    print(A @ B)
    print(A.T)
    print(np.dot(A, B))
    mat_a = np.array([1, 2, 3])
    mat_b = np.array([4, 5, 6])
    print("-->", cosine_similarity(mat_a, mat_b))
    mat_A = np.array([[2, 0], [0, 3]])
    eigen_values, eigen_vectors = np.linalg.eig(mat_A)
    print("Eigen Values: ",eigen_values)
    print("Eigen Vector: ", eigen_vectors)
    mat_B = np.array([[4, 1], [2, 3]])
    eigen_values_b, eigen_vectors_b = np.linalg.eig(mat_B)
    print("Eigen Values b: ", eigen_values_b)
    print("Eigen Vector b: ", eigen_vectors_b)
    v = np.array([3, -4])
    l1 = np.linalg.norm(v, ord=1)
    l2 = np.linalg.norm(v, ord=2)
    print("l1: ", l1)
    print("l2: ", l2)


