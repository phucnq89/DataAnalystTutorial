# %% Import library
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# %% Load data
df = pd.read_excel('./data/Sales.xlsx')

# %% Create model
x = df[['Price', 'Ads_Cost']]
y = df[['Sales_Volume']]
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
x_new = [[4.2, 4.0], [4.8, 4.3]]
predicted_future_values = model.predict(x_new)

# %% Split train/test dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# %% Create new model and fit it
new_model = LinearRegression().fit(x_train, y_train)
score = new_model.score(x_train, y_train)
print("R2: ", score)
print(y_test)
y_pred = new_model.predict(x_test)
print(y_pred)