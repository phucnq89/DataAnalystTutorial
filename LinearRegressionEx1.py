# %% Import library
from scipy import stats
import matplotlib.pyplot as plt

# %% Prepare data
income = [8,9,10,11,12,15,15,16,17,18,18,20,24,25,25]
expenditure = [6,7,9,9,10,12,11,13,13,15,14,16,18,20,19]

# %% Create model and fit it
slope, intercept, r, p, std_err = stats.linregress(income, expenditure)
def func(x):
    return intercept + slope * x
model = list(map(func, income))

plt.scatter(income, expenditure)
plt.xlabel("Income")
plt.ylabel("Expenditure")
plt.plot(income, model)
plt.show()

# %% Predict response
predicted_values = []
for i in income:
    predicted_values.append(func(i))

plt.plot(income, expenditure, color='b', label="Actual")
plt.plot(income, predicted_values, color='r',label="Predict", linestyle='--', marker='x')
plt.xlabel("Income")
plt.ylabel("Expenditure")
plt.legend()
plt.grid(True)
plt.show()

# %% Predict for future values
# pre_exp = func(23)
