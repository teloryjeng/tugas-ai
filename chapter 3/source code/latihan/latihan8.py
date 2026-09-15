from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
total_points = y_test.shape[0]
correct_points = (y_test == y_pred).sum()

print("Wine Dataset - Total points: %d | Correctly labeled: %d (Akurasi: %.2f%%)" % 
      (total_points, correct_points, (correct_points / total_points) * 100))