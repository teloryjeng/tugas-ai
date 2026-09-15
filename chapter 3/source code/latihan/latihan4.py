import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)

selected_features = ['mean radius', 'mean texture', 'mean area', 'mean smoothness']
df[selected_features].hist(bins=20, figsize=(10, 8))
plt.suptitle('Histogram Fitur Breast Cancer')
plt.tight_layout()
plt.show()