import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV

# Setup models
# For the model, include the Y-intercept and force positive coefficients
linear_regressor = LinearRegression(fit_intercept=True, positive=True)
# Elastic is same as Ridge when l1_ratio=0
ridge_regressor = ElasticNetCV(fit_intercept=True, positive=True,
                               tol=1e-5, max_iter=10000,
                               l1_ratio=0, alphas=np.arange(0.001, 0.1, 0.001))
elastic_regressor = ElasticNetCV(fit_intercept=True, positive=True,
                                 l1_ratio=np.arange(0.01, 0.3, 0.01),
                                 tol=1e-5, max_iter=10000)

# Setup variables and boat parameters
# See google sheet document:
# https://docs.google.com/spreadsheets/d/1wlKaxpww5Z7icWqO8fdrtR49Uo22-nshw6ypt9hyWGg/edit?usp=sharing
fuel_gauge_range = 16.0  # fuel gauge range in gallons
variables = ['Trick Sets', 'Slalom Sets <= 30mph', 'Slalom Sets > 30mph', 'Jump Sets <= 28mph', 'Jump Sets > 28mph']
# Ski sets to estimate gas for, same order as variables:
to_predict = np.array([[4, 2, 2, 0, 0]])

# Setup Data
df = pd.read_csv('data.csv', sep=',', header=0)
df['Change'] = df['Start Reading'] - df['End Reading']
df['Gallons'] = df['Change']*fuel_gauge_range
y = df['Gallons'].values
X = df[variables].values
sample_weights = df['Weight'].values

# Make models
# TODO: make models without weights once we have more data
linear_model = linear_regressor.fit(X, y, sample_weights)
elastic_model = elastic_regressor.fit(X, y, sample_weights)
ridge_model = ridge_regressor.fit(X, y, sample_weights)

print(f"Average gallons of gas used: {pd.Series.mean(df['Gallons'])}")
print()
print('Linear model:')
print(f'coefficients: {linear_model.coef_}')
print(f'intercept: {linear_model.intercept_}')
print(f'R^2: {linear_model.score(X, y, sample_weights)}')
print(f'prediction: {linear_model.predict(to_predict)}')
print()
print('Ridge model:')
print(f'coefficients: {ridge_model.coef_}')
print(f'intercept: {ridge_model.intercept_}')
print(f'alpha: {ridge_model.alpha_}')
print(f'R^2: {ridge_model.score(X, y, sample_weights)}')
print(f'prediction: {ridge_model.predict(to_predict)}')
print()
print('Elastic model:')
print(f'coefficients: {elastic_model.coef_}')
print(f'intercept: {elastic_model.intercept_}')
print(f'l1 ratio: {elastic_model.l1_ratio_}')
print(f'alpha: {elastic_model.alpha_}')
print(f'R^2: {elastic_model.score(X, y, sample_weights)}')
print(f'prediction: {elastic_model.predict(to_predict)}')
