from sklearn.datasets import load_breast_cancer
import pandas as pd
from pycaret import classification

data = load_breast_cancer(as_frame=True)
df = data.frame

classification.setup(data=df, target='target', session_id=42)
best_model = classification.compare_models()
print(best_model)