from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
import joblib

X, y = load_iris(return_X_y=True)

# Latih model
clf = GaussianNB()
clf.fit(X, y)

# Simpan model ke file
joblib.dump(clf, 'naive_bayes_model.pkl')
print("Model berhasil disimpan ke naive_bayes_model.pkl")

# Muat kembali model dari file
clf_loaded = joblib.load('naive_bayes_model.pkl')

# Lakukan prediksi
sample = [[5.0, 3.4, 1.5, 0.4]]
p = clf_loaded.predict(sample)
print("Hasil prediksi dari model yang dimuat:", p)