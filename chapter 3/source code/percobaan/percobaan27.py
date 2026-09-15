# Example 3.27 The PyCaret_demo.py
import pandas as pd
from sklearn import datasets

# 1. Load Iris (tanpa parameter as_frame)
iris_raw = datasets.load_iris()

# 2. Buat DataFrame secara manual
iris = pd.DataFrame(iris_raw.data, columns=iris_raw.feature_names)
iris['Target'] = iris_raw.target

print(iris.head())

# 3. PyCaret Setup & Compare Models
from pycaret import classification
classification.setup(data=iris, target='Target')
classification.compare_models()
