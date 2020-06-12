import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn import linear_model

import parametric_lasso
import gen_data
import util


def run():
    n = 100
    p = 5
    lamda = 1
    beta_vec = [2, 2, 0, 0, 0]
    cov = np.identity(n)

    threshold = 20

    X, y, true_y = gen_data.generate(n, p, beta_vec)

    clf = linear_model.Lasso(alpha=lamda, fit_intercept=False, normalize=False)
    clf.fit(X, y)
    bh = clf.coef_

    y = y.reshape((n, 1))
    true_y = true_y.reshape((n, 1))

    A, XA, Ac, XAc, bhA = util.construct_A_XA_Ac_XAc_bhA(X, bh, n, p)

    if len(A) == 0:
        return None

    y = y.reshape((n, 1))

    rand_value = np.random.randint(len(A))
    j_selected = A[rand_value]

    etaj, etajTy = util.construct_test_statistic(j_selected, XA, y, A)

    list_zk, list_bhz, list_active_set = parametric_lasso.run_parametric_lasso(X, y, lamda, etaj, n, p, threshold)

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
    plt.plot(grid, sm.distributions.ECDF(np.array(list_pivot))(grid), 'r-', linewidth=5, label='Pivot Lasso TN-A')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./results/pivot_lasso_TN_A.png', dpi=100)
    plt.show()