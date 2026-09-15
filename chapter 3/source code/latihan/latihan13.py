from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Buat dataset klaster
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

kmeans = KMeans(n_clusters=3, random_state=0)
y_kmeans = kmeans.fit_predict(X)

print("Pusat Klaster:\n", kmeans.cluster_centers_)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=30, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c='red', marker='X', label='Centroids')
plt.title('K-Means Clustering dengan make_blobs')
plt.legend()
plt.show()