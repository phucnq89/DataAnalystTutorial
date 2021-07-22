# %% Import Library
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# %% Load data
df = pd.read_csv('data/Income.csv')
Income = df["Income"]
Expenditure = df["Expenditure"]
Income, Expenditure = np.array(Income), np.array(Expenditure)
Income = sm.add_constant(Income)

# %% Create model
model = sm.OLS(Expenditure, Income)
results = model.fit()

# %% Get results
R_sq = results.rsquared
params = results.params
print(results.summary())

plt.scatter(df["Income"], df["Expenditure"])
plt.plot(df["Income"], params[0] + params[1] * df["Income"])
plt.show()

# %% Predict for present values
predicted_values = results.predict(Income)
plt.plot(df["Income"], df["Expenditure"], color='b', label="Actual")
plt.plot(df["Income"], predicted_values, color='r',label="Predict", linestyle='--', marker='x')
plt.xlabel("Income")
plt.ylabel("Expenditure")
plt.legend()
plt.grid(True)
plt.show()

# %% Predict for future values
income_new = np.array([23, 26, 28])
income_new = sm.add_constant(income_new)
predicted_future_values = results.predict(income_new)