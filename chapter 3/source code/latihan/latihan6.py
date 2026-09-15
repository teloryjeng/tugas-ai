from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X, y = make_classification(n_samples=2000, n_features=6,
                           n_informative=4, n_redundant=0,
                           random_state=0, shuffle=False)

clf = LinearDiscriminantAnalysis()
clf.fit(X, y)

test_sample = [[0, 0, 0, 0, 0, 0]]
pred = clf.predict(test_sample)
print("Dimensi X:", X.shape)
print("Hasil Prediksi:", pred)