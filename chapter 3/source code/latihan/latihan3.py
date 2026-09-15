import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
df = df.dropna()

species = df['species'].unique()
colors = ['red', 'green', 'blue']

plt.figure(figsize=(8, 6))
for sp, color in zip(species, colors):
    subset = df[df['species'] == sp]
    plt.scatter(subset['sepal_length'], subset['sepal_width'], color=color, label=sp)

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Scatter Plot Iris: Sepal Length vs Sepal Width')
plt.legend()
plt.grid(True)
plt.show()