import numpy as np

def linear_reqression(X, y):
    '''
    X = [[9, 1, 4.1, 5....],
         [1, 1.3, 47.1, 4.8 ....],
         [5, 1.2, 4.1, 3....],
         [3, 1.3, 9, 9....],...]

    y = [[9, 1, 4.1, 5....],
         [1, 1.3, 47.1, 4.8 ....],
         [5, 1.2, 4.1, 3....],
         [3, 1.3, 9, 9....],...]'''
    X_mean = X.mean()
    y_mean = y.mean()
    w = np.sum((X - X_mean) * (y - y_mean)) / np.sum(np.square(X - X_mean))
    b = y_mean - w * X_mean
    return w, b

k = 4.5
n = 10
X = np.arange(1000)[:,np.newaxis]
y = X * k + n + np.random.randn(*X.shape)

print(linear_reqression(X, y))