from sklearn.datasets import load_linnerud
from sklearn.linear_model import LinearRegression

linnerud = load_linnerud()
X = linnerud.data    # Chins, Situps, Jumps
y = linnerud.target  # Weight, Waist, Pulse

reg = LinearRegression()
reg.fit(X, y)

print("Fitur:", linnerud.feature_names)
print("Target:", linnerud.target_names)
print("Koefisien:\n", reg.coef_)
print("Intercept:\n", reg.intercept_)

# Prediksi untuk satu baris data latihan
pred = reg.predict([X[0]])
print("Prediksi untuk sampel pertama:", pred)
print("Nilai sebenarnya:", y[0])