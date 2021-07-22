# %% - Import library
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

# %% - Read prepared data
df = pd.read_csv('data/cars_scale.csv')
x = df[['Weight', 'Volume']]
scaleX = scale.fit_transform(x)
y = df['CO2']

# %% - Create model and fit it
model = linear_model.LinearRegression().fit(scaleX, y)

# %% - Get results
r_sq = model.score(scaleX, y)
print('Cofficient of determination: ', r_sq)
print('intercept: ', model.intercept_)
print('slope: ', model.coef_)

# %% - Predict response
"""predict the CO2 emission of a car where
the weight is 3300kg, and the volume is 1.3l:"""
scaled_value = scale.transform([[3300, 1.3]])
predictedCO2 = model.predict([scaled_value[0]])