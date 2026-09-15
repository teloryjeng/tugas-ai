from sklearn import svm, datasets

iris = datasets.load_iris()
# Mengambil fitur ke-3 dan ke-4 (indeks kolom 2 dan 3)
X = iris.data[:, 2:4]
y = iris.target

clf = svm.SVC()
clf.fit(X, y)

# Prediksi menggunakan fitur petal length dan petal width (contoh: 4.5 cm dan 1.5 cm)
p = clf.predict([[4.5, 1.5]])
print("Hasil Prediksi dengan fitur Petal:", p)