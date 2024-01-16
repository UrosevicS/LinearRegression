import numpy as np


def linear_reqression(X, y, epochs=10000, learning_rate=0.01):
    '''
    X = [[9, 1, 4.1, 5....],
         [1, 1.3, 47.1, 4.8 ....],
         [5, 1.2, 4.1, 3....],
         [3, 1.3, 9, 9....],...]

    y = [[9, 1, 4.1, 5....],
         [1, 1.3, 47.1, 4.8 ....],
         [5, 1.2, 4.1, 3....],
         [3, 1.3, 9, 9....],...]'''
    ones = np.ones((1, X.shape[1]))
    X = np.append(ones, X, axis=0)
    w = np.zeros((2, 1))

    for epoch in range(epochs):
        y_pred = np.dot(w.T, X)

        dw = 2 / X.shape[1] * np.dot(X, (y_pred - y).T)
        w -= dw * learning_rate
    return w


k = 4.5
n = 10
X = np.arange(1000)[np.newaxis, :] / 1000
y = X * k + n + np.random.randn(*X.shape)

print(linear_reqression(X, y))
