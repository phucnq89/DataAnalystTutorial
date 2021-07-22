# %% Import library
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# %% Load data
df = pd.read_csv('./data/cars.csv')

# %% Create model
x = df[['Weight', 'Volume']]
y = df[['CO2']]
model = LinearRegression().fit(x, y)

# %% Get results
R_sq = model.score(x, y)
intercept = model.intercept_
coefs = model.coef_

# %% Predict for present values
predicted_values = model.predict(x)
plt.plot(y, label='Actual')
plt.plot(predicted_values, label="Predict", marker='x', color='darkgreen', linestyle='--')
plt.legend()
plt.show()

# %% Predict for future values
x_new = [[1500, 1300], [1400, 1100]]
predicted_future_values = model.predict(x_new)