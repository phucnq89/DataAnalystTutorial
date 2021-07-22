# %% Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# %% Load data
df = pd.read_csv('./data/HomePrices2.csv')
x = df[['Area', 'Bedrooms']]
y = df['Price']
x, y = np.array(x), np.array(y)
x = sm.add_constant(x)

# %% Visualize data
plt.scatter(df.Price, df.Bedrooms)
plt.show()

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
plt.plot(df["Price"], label="Actual")
plt.plot(predicted_values, label="Predict", color='r', marker='x', linestyle='--')
plt.ylabel("Home prices")
plt.legend()
plt.grid(True)
plt.show()

# %% Predict for future values
x_new = np.array([[75, 2.5], [110, 3]])
x_new = sm.add_constant(x_new)
predicted_future_values = results.predict(x_new)

