import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def svd_scratch():
    try:
        image_color = mpimg.imread("demoImage.png")
    except FileNotFoundError:
        print("File not found")
        exit()
    if len(image_color.shape) == 3:
        img = np.dot(image_color[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        img = image_color
    U, S, Vt = np.linalg.svd(img, full_matrices=False)
    k = 50
    compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original")
    plt.subplot(2, 2, 2)
    plt.imshow(compressed, cmap="gray")
    plt.title(f"Compressed (k={k})")
    plt.savefig("compressed_updated.png")
    plt.show()

def pca_scratch(X, n_components):
    X_centered = X - X.mean(axis=0)
    covariance_matrix = np.cov(X_centered, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    order = np.argsort(eigen_values)[::-1]
    top_vectors = eigen_vectors[:, order[:n_components]]
    return X_centered @ top_vectors

def pca_with_database():
    df = pd.read_csv("dataset_for_pca.csv")
    numeric_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "sepal_area"]
    X = df[numeric_columns].values
    label = df["species"].values
    X_reduced = pca_scratch(X, n_components=2)
    plt.figure(figsize=(8, 6))
    for spices in np.unique(label):
        mask = label == spices
        plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1], label=spices, alpha=0.8, edgecolors="k")
    plt.title("PCA of Iris Dataset (5D --> 2D)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("pca.png")
    plt.show()

if __name__ == "__main__":
    svd_scratch()
    pca_with_database()