import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV

# Setup models
# For the model, include the Y-intercept and force positive coefficients
linear_regressor = LinearRegression(fit_intercept=True, positive=True)
# same as ridgecv when l1_ratio=0
elastic_regressor = ElasticNetCV(fit_intercept=True, positive=True,
                                 tol=1e-5, max_iter=10000,
                                 l1_ratio=0, alphas=np.arange(0.001, 0.1, 0.001))
# Setup variables and boat parameters
fuel_gauge_range = 16.0  # fuel gauge range in gallons
variables = ['Trick Sets', 'Slalom Sets <= 30mph', 'Slalom Sets > 30mph', 'Jump Sets <= 28mph', 'Jump Sets > 28mph']
# Ski sets to estimate gas for, same order as variables:
to_predict = np.array([[4, 2, 2, 0, 0]])

# Setup Data
df = pd.read_csv('data.csv', sep=',', header=0)
df['Change'] = df['Starting Reading'] - df['Ending Reading']
y = df['Change'].values
X = df[variables].values
sample_weights = df['Weight'].values

# Make models
linear_model_weighted = linear_regressor.fit(X, y, sample_weights)
elastic_model_weighted = elastic_regressor.fit(X, y, sample_weights)

print('Linear weighted coefficients, intercept, R^2, prediction:')
print(linear_model_weighted.coef_)
print(linear_model_weighted.intercept_)
print(linear_model_weighted.score(X, y, sample_weights))
print(linear_model_weighted.predict(to_predict) * fuel_gauge_range)

print('Elastic weighted coefficients, intercept, l1 ratio, alpha, R^2, prediction:')
print(elastic_model_weighted.coef_)
print(elastic_model_weighted.intercept_)
print(elastic_model_weighted.l1_ratio_)
print(elastic_model_weighted.alpha_)
print(elastic_model_weighted.score(X, y, sample_weights))
print(elastic_model_weighted.predict(to_predict) * fuel_gauge_range)
