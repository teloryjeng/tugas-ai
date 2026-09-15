from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X, y = load_diabetes(return_X_y=True)

# Target regresi diubah menjadi biner agar dapat diklasifikasikan
y_class = (y > np.median(y)).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.5, random_state=0)

clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
total_points = y_test.shape[0]
correct_points = (y_test == y_pred).sum()

print("Diabetes Classification - Total: %d | Correct: %d (Akurasi: %.2f%%)" % 
      (total_points, correct_points, (correct_points / total_points) * 100))