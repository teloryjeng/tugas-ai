import pandas as pd
import warnings
warnings.filterwarnings('ignore') # Menyembunyikan pesan warning agar terminal tidak penuh

from auto_ml import Predictor
from sklearn.datasets import fetch_california_housing

# 1. Load Data (menggunakan seluruh data California Housing)
data = fetch_california_housing()

# 2. Buat DataFrame secara manual
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target

# 3. Split Dataset (80% latih, 20% uji)
df_train = df.sample(frac=0.8, random_state=42)
df_test = df.drop(df_train.index)

# 4. Pelatihan dengan auto_ml
column_descriptions = {
    'MedHouseVal': 'output'
}

ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

# Proses pelatihan auto_ml (tunggu sekitar 2-3 menit hingga selesai)
ml_predictor.train(df_train)

# Evaluasi skor akhir
ml_predictor.score(df_test, df_test.MedHouseVal)
