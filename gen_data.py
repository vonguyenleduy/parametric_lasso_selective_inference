import numpy as np


def generate(n, p, beta_vec):
    X = []
    y = []
    true_y = []

    for i in range(n):
        X.append([])
        yi = 0
        for j in range(p):
            xij = np.random.normal(0, 1)
            X[i].append(xij)
            yi = yi + xij * beta_vec[j]

        true_y.append(yi)
        noise = np.random.normal(0, 1)
        yi = yi + noise
        y.append(yi)

    X = np.array(X)
    y = np.array(y)
    true_y = np.array(true_y)

    return X, y, true_y


def generate_non_normal(n, p, beta_vec):
    X = []
    y = []
    true_y = []

    for i in range(n):
        X.append([])
        yi = 0
        for j in range(p):
            xij = np.random.normal(0, 1)
            X[i].append(xij)
            yi = yi + xij * beta_vec[j]

        true_y.append(yi)

        noise = np.random.normal(0, 1)
        # noise = np.random.laplace(0, 1)
        # noise = skewnorm.rvs(a=10, loc=0, scale=1)
        # noise = np.random.standard_t(20)

        yi = yi + noise
        y.append(yi)

    X = np.array(X)
    y = np.array(y)
    true_y = np.array(true_y)

    return X, y, true_y