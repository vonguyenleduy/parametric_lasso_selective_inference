import numpy as np
from sklearn import linear_model

import parametric_lasso
import gen_data
import util
import ci


def run():
    alpha = 0.05
    n = 100
    p = 5
    lamda = 0.05
    beta_vec = [1, 1, 0, 0, 0]
    cov = np.identity(n)

    threshold = 20

    X, y, _ = gen_data.generate(n, p, beta_vec)

    clf = linear_model.Lasso(alpha=lamda, fit_intercept=False, normalize=False)
    clf.fit(X, y)
    bh = clf.coef_

    y = y.reshape((n, 1))

    A, XA, Ac, XAc, bhA = util.construct_A_XA_Ac_XAc_bhA(X, bh, n, p)

    if len(A) == 0:
        return None

    for j_selected in A:
        etaj, etajTy = util.construct_test_statistic(j_selected, XA, y, A)

        list_zk, list_bhz, list_active_set = parametric_lasso.run_parametric_lasso(X, y, lamda, etaj, n, p, threshold)
        confidence_interval = ci.compute_ci(A, bh, list_active_set, list_zk, list_bhz, etaj, etajTy, cov,
                                            beta_vec[j_selected], alpha)

        print('Feature', j_selected + 1, ' True Beta:', beta_vec[j_selected],
              ' CI: ' + '[{:.2f}'.format(confidence_interval[0]) + ', {:.2f}]'.format(confidence_interval[1]),
              ' CI Length', '{:.2f}'.format(confidence_interval[1] - confidence_interval[0]))
        print("==========")


if __name__ == '__main__':
    run()