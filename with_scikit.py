import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

k = 4.5
n = 10
X = np.arange(1000)[:,np.newaxis]
y = X * k + n + np.random.randn(*X.shape)

model = LinearRegression()

model.fit(X, y)
y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)

print(mse)
print(model.coef_, model.intercept_del)
