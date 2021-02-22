import numpy as np
from sklearn.linear_model import ElasticNet
import util


def compute_quotient(numerator, denominator):
    if denominator == 0:
        return np.Inf

    quotient = numerator / denominator

    if quotient <= 0:
        return np.Inf

    return quotient


def parametric_elastic_net(X, yz, lamda, delta, b, n, p):
    alpha_for_elastic_net = lamda + delta
    l1_ratio_for_elastic_net = lamda / (lamda + delta)

    regr = ElasticNet(alpha=alpha_for_elastic_net, l1_ratio=l1_ratio_for_elastic_net,
                      fit_intercept=False, normalize=False, tol=1e-10)

    regr.fit(X, yz)
    bhz = regr.coef_

    Az, XAz, Acz, XAcz, bhAz = util.construct_A_XA_Ac_XAc_bhA(X, bhz, n, p)

    etaAz = np.array([])

    if XAz is not None:
        inv = np.linalg.pinv(np.dot(XAz.T, XAz) + n * delta * np.identity(len(Az)))
        invXAzT = np.dot(inv, XAz.T)
        etaAz = np.dot(invXAzT, b)

    shAz = np.array([])
    gammaAz = np.array([])

    if XAcz is not None:
        if XAz is None:
            e1 = yz
        else:
            e1 = yz - np.dot(XAz, bhAz)

        e2 = np.dot(XAcz.T, e1)
        shAz = e2/(lamda * n)

        if XAz is None:
            gammaAz = (np.dot(XAcz.T, b)) / n
        else:
            gammaAz = (np.dot(XAcz.T, b) - np.dot(np.dot(XAcz.T, XAz), etaAz)) / n

    bhAz = bhAz.flatten()
    etaAz = etaAz.flatten()
    shAz = shAz.flatten()
    gammaAz = gammaAz.flatten()

    min1 = np.Inf
    min2 = np.Inf

    for j in range(len(etaAz)):
        numerator = - bhAz[j]
        denominator = etaAz[j]

        quotient = compute_quotient(numerator, denominator)

        if quotient < min1:
            min1 = quotient

    for j in range(len(gammaAz)):
        numerator = (np.sign(gammaAz[j]) - shAz[j])*lamda
        denominator = gammaAz[j]

        quotient = compute_quotient(numerator, denominator)

        if quotient < min2:
            min2 = quotient

    return min(min1, min2), Az, bhz


def run_parametric_elastic_net(X, y, lamda, delta, etaj, n, p, threshold):

    zk = -threshold

    list_zk = [zk]
    list_active_set = []
    list_bhz = []

    while zk < threshold:
        yz, b = util.compute_yz(y, etaj, zk, n)

        skz, Akz, bhkz = parametric_elastic_net(X, yz, lamda, delta, b, n, p)
        zk = zk + skz + 0.001

        if zk < threshold:
            list_zk.append(zk)
        else:
            list_zk.append(threshold)

        list_active_set.append(Akz)
        list_bhz.append(bhkz)

    return list_zk, list_bhz, list_active_set