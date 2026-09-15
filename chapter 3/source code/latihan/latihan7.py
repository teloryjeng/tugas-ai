import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Terapkan PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Explained variance ratio:", pca.explained_variance_ratio_)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolors='k')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA pada Dataset Breast Cancer')
plt.colorbar(scatter, ticks=[0, 1], label='Kelas (0: Malignant, 1: Benign)')
plt.show()