import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
import statsmodels.api as sm

import gen_data
import util
import parametric_elastic_net


def run():
    n = 100
    p = 5
    lamda = 0.5
    delta = 0.5
    threshold = 20
    beta_vec = [2, 2, 0, 0, 0]
    cov = np.identity(n)

    X, y, true_y = gen_data.generate(n, p, beta_vec)

    y = y.reshape((n, 1))
    true_y = true_y.reshape((n, 1))

    alpha_for_elastic_net = lamda + delta
    l1_ratio_for_elastic_net = lamda / (lamda + delta)

    regr = ElasticNet(alpha=alpha_for_elastic_net, l1_ratio=l1_ratio_for_elastic_net,
                      fit_intercept=False, normalize=False, tol=1e-10)

    regr.fit(X, y)
    bh = regr.coef_

    A, XA, Ac, XAc, bhA = util.construct_A_XA_Ac_XAc_bhA(X, bh, n, p)

    rand_value = np.random.randint(len(A))
    j_selected = A[rand_value]

    etaj, etajTy = util.construct_test_statistic(j_selected, XA, y, A)

    list_zk, list_bhz, list_active_set = parametric_elastic_net.run_parametric_elastic_net(X, y, lamda, delta, etaj, n, p, threshold)

    tn_mu = np.dot(etaj.T, true_y)[0][0]
    pivot = util.pivot(A, bh, list_active_set, list_zk, list_bhz, etaj, etajTy, cov, tn_mu, 'A')
    return pivot


if __name__ == '__main__':

    max_iteration = 1200
    list_pivot = []

    for each_iter in range(max_iteration):
        print(each_iter)
        pivot = run()
        if pivot is not None:
            list_pivot.append(pivot)

    plt.rcParams.update({'font.size': 18})
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, sm.distributions.ECDF(np.array(list_pivot))(grid), 'r-', linewidth=6, label='Pivot Elastic Net TN-A')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.savefig('./results/pivot_elastic_net_TN_A.png', dpi=100)
    plt.show()

