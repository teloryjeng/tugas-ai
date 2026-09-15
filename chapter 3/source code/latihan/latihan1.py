from sklearn import svm

# Menambahkan 2 sampel tambahan (total 6 sampel)
X = [
    [170, 70, 10], 
    [180, 80, 12], 
    [170, 65, 8], 
    [160, 55, 7],
    [175, 75, 11],  # Sampel 5: Laki-laki
    [155, 50, 6]    # Sampel 6: Perempuan
]
y = [0, 0, 1, 1, 0, 1]

clf = svm.SVC()
clf.fit(X, y)

p = clf.predict([[160, 60, 7]])
print("Hasil Prediksi (0: Laki-laki, 1: Perempuan):", p)