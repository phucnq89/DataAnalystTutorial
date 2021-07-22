# %% Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# %% Load data
df = pd.read_excel('./data/Sales.xlsx')
x = df[['Price', 'Ads_Cost']]
y = df['Sales_Volume']
x, y = np.array(x), np.array(y)
x = sm.add_constant(x)

# %% Create model
model = sm.OLS(y, x)
results = model.fit()
#
# %% Get resuts
R_sq = results.rsquared
params = results.params
print(results.summary())

# %% Predict for present values
predicted_values = results.predict(x)
plt.plot(df["Sales_Volume"], label="Actual")
plt.plot(predicted_values, label="Predict", color='r', marker='x', linestyle='--')
plt.ylabel("Sales Volume")
plt.legend()
plt.grid(True)
plt.show()

# %% Predict for future values
x_new = np.array([[4.2, 4.0], [4.8, 4.3]])
x_new = sm.add_constant(x_new)
predicted_future_values = results.predict(x_new)

