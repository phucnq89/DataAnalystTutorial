# %% Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# %% Load data
df = pd.read_csv('./data/cars.csv')
x = df[['Weight', 'Volume']]
y = df['CO2']
x, y = np.array(x), np.array(y)
x = sm.add_constant(x)

# %% Create model
model = sm.OLS(y, x)
results = model.fit()

# %% Get resuts
R_sq = results.rsquared
params = results.params
print(results.summary())

# %% Predict for present values
predicted_values = results.predict(x)
plt.plot(df["CO2"], label="Actual")
plt.plot(predicted_values, label="Predict", color='r', marker='x', linestyle='--')
plt.ylabel("CO2")
plt.legend()
plt.grid(True)
plt.show()

# %% Predict for future values
x_new = np.array([[1500, 1300], [1400, 1100]])
x_new = sm.add_constant(x_new)
predicted_future_values = results.predict(x_new)

