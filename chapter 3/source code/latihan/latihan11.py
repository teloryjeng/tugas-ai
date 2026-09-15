import matplotlib.pyplot as plt
from scipy import stats

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [2, 3, 5, 5, 7, 8, 9, 10, 12, 13]

slope, intercept, r, p, std_err = stats.linregress(x, y)
print("Slope: ", slope)
print("Intercept: ", intercept)

def myfunc(val):
    return slope * val + intercept

mymodel = list(map(myfunc, x))

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue', label='Data Poin')
plt.plot(x, mymodel, color='red', label=f'Fit Line: y={slope:.2f}x+{intercept:.2f}')
plt.title('Regresi Linear Sederhana')
plt.xlabel('Sumbu X')
plt.ylabel('Sumbu Y')
plt.legend()
plt.grid(True)
plt.show()