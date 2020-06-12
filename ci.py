import numpy as np
from mpmath import mp

mp.dps = 500


def f(z_interval, etajTy, mu, tn_sigma):
    numerator = 0
    denominator = 0

    for each_interval in z_interval:
        al = each_interval[0]
        ar = each_interval[1]

        denominator = denominator + mp.ncdf((ar - mu) / tn_sigma) - mp.ncdf((al - mu) / tn_sigma)

        if etajTy >= ar:
            numerator = numerator + mp.ncdf((ar - mu) / tn_sigma) - mp.ncdf((al - mu) / tn_sigma)
        elif (etajTy >= al) and (etajTy < ar):
            numerator = numerator + mp.ncdf((etajTy - mu) / tn_sigma) - mp.ncdf((al - mu) / tn_sigma)

    if denominator != 0:
        return float(numerator / denominator)
    else:
        return np.Inf


def find_root(z_interval, etajTy, tn_sigma, y, lb, ub, tol=1e-6):
    a, b = lb, ub
    fa, fb = f(z_interval, etajTy, a, tn_sigma), f(z_interval, etajTy, b, tn_sigma)

    # assume a < b
    if (fa > y) and (fb > y):
        while fb > y:
            b = b + (b - a)
            fb = f(z_interval, etajTy, b, tn_sigma)
    elif (fa < y) and (fb < y):
        while fa < y:
            a = a - (b - a)
            fa = f(z_interval, etajTy, a, tn_sigma)

    max_iter = int(np.ceil((np.log(tol) - np.log(b - a)) / np.log(0.5)))

    c = None
    for _ in range(max_iter):
        c = (a + b) / 2
        fc = f(z_interval, etajTy, c, tn_sigma)
        if fc > y:
            a = c
        elif fc < y:
            b = c

    return c


def equal_tailed_interval(z_interval, etajTy, alpha, tn_mu, tn_sigma):
    lb = tn_mu - 20. * tn_sigma
    ub = tn_mu + 20. * tn_sigma

    L = find_root(z_interval, etajTy, tn_sigma, 1.0 - 0.5 * alpha, lb, ub)
    U = find_root(z_interval, etajTy, tn_sigma, 0.5 * alpha, lb, ub)

    return np.array([L, U])


def compute_ci(A, bh, list_active_set, list_zk, list_bhz, etaj, etajTy, cov, tn_mu, alpha):
    tn_sigma = np.sqrt(np.dot(np.dot(etaj.T, cov), etaj))[0][0]

    z_interval = []

    for i in range(len(list_active_set)):
        if np.array_equal(A, list_active_set[i]):
            z_interval.append([list_zk[i], list_zk[i + 1] - 1e-10])

    new_z_interval = []

    for each_interval in z_interval:
        if len(new_z_interval) == 0:
            new_z_interval.append(each_interval)
        else:
            sub = each_interval[0] - new_z_interval[-1][1]
            if abs(sub) < 0.01:
                new_z_interval[-1][1] = each_interval[1]
            else:
                new_z_interval.append(each_interval)

    z_interval = new_z_interval

    ci = equal_tailed_interval(z_interval, etajTy, alpha, tn_mu, tn_sigma)

    return ci


def compute_ci_with_specified_interval(z_interval, etaj, etajTy, cov, tn_mu, alpha):
    tn_sigma = np.sqrt(np.dot(np.dot(etaj.T, cov), etaj))[0][0]

    ci = equal_tailed_interval(z_interval, etajTy, alpha, tn_mu, tn_sigma)

    return ci