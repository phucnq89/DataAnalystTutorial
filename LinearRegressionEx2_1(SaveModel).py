# %% Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# %% Load data
df = pd.read_csv('data/Income.csv')
x = df[['Income']]
y = df[['Expenditure']]

# %% Create model
model = LinearRegression().fit(x, y)

# %% Get results
R_sq = model.score(x, y)
intercept = model.intercept_
slope = model.coef_

plt.scatter(x, y)
plt.xlabel("Income")
plt.ylabel("Expenditure")
plt.plot(x, intercept + slope * x)
plt.show()

# %% Predict for present values
predicted_values = model.predict(x)
# predicted_values = intercept + slope * x
plt.plot(x, y, color='b', label="Actual")
plt.plot(x, predicted_values, color='r',label="Predict", linestyle='--', marker='x')
plt.xlabel("Income")
plt.ylabel("Expenditure")
plt.legend()
plt.grid(True)
plt.show()

# %% Predict for future values
future_income = np.array([23, 26, 28]).reshape(-1,1)
predicted_future_values = model.predict(future_income)
# predicted_future_values = intercept + slope * future_income

# %% Save model for prediction (using Pickle)
import pickle
with open('./models/model_pickle', 'wb') as f:
    pickle.dump(model, f)

with open('./models/model_pickle', 'rb') as f:
    model_p = pickle.load(f)
pred_values = model_p.predict(future_income)
