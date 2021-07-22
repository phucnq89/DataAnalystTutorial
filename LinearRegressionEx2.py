# %% Import library
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# %% Prepare data
income = np.array([8,9,10,11,12,15,15,16,17,18,18,20,24,25,25]).reshape((-1, 1))
expenditure = np.array([6,7,9,9,10,12,11,13,13,15,14,16,18,20,19])

# %% Create model
model = LinearRegression().fit(income, expenditure)

# %% Get results
R_sq = model.score(income, expenditure)
intercept = model.intercept_
slope = model.coef_

plt.scatter(income, expenditure)
plt.xlabel("Income")
plt.ylabel("Expenditure")
plt.plot(income, intercept + slope * income)
plt.show()

# %% Predict for present values
predicted_values = intercept + slope * income
plt.plot(income, expenditure, color='b', label="Actual")
plt.plot(income, predicted_values, color='r',label="Predict", linestyle='--', marker='x')
plt.xlabel("Income")
plt.ylabel("Expenditure")
plt.legend()
plt.grid(True)
plt.show()

# %% Predict for future values
# income_new = np.array([23, 26, 28]).reshape(-1, 1)
# predicted_future_values = intercept + slope * income_new