import numpy as np
from sklearn.semi_supervised import LabelSpreading

# 3 Kelompok: Titik sekitar (0,0), sekitar (10,5), dan sekitar (20,10)
X = np.array([
    [0, 1], [1, 1], [2, 0],       # Kelompok 0
    [10, 5], [11, 6], [12, 4],    # Kelompok 1
    [20, 10], [21, 11], [22, 9]   # Kelompok 2
])

# Total 9 sampel
labels = np.full(9, -1.)
labels[0] = 0  # 1 label untuk kelompok 0
labels[3] = 1  # 1 label untuk kelompok 1
labels[6] = 2  # 1 label untuk kelompok 2

print("Label sebelum spreading:")
print(labels)

label_spread = LabelSpreading(kernel='knn', alpha=0.8)
label_spread.fit(X, labels)

print("\nLabel sesudah spreading:")
print(label_spread.transduction_)