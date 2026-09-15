import numpy as np
from sklearn.semi_supervised import LabelSpreading

# 4 titik awal grup 1 + 2 titik baru, 4 titik awal grup 2 + 2 titik baru (total 12 titik)
X = np.array([
    [0, 1], [1, 1], [2, 0], [3, 1], [1, 2], [2, 2],         # Grup 1
    [10, 5], [11, 6], [12, 4], [13, 5], [11, 4], [12, 6]     # Grup 2
])

# Label awal: hanya titik pertama dan titik ke-7 yang diberi label
labels = np.full(12, -1.)
labels[0] = 0   # Grup 1
labels[6] = 1   # Grup 2

print("Label sebelum spreading:")
print(labels)

label_spread = LabelSpreading(kernel='knn', alpha=0.8)
label_spread.fit(X, labels)

output_labels = label_spread.transduction_
print("\nLabel sesudah spreading:")
print(output_labels)