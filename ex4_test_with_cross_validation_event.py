import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn import linear_model

import parametric_lasso
import gen_data
import util



def check_zero(value):
    if -1e-12 <= value <= 1e-12:
        return 0

    return value


def compute_a_b(y, etaj, n):
    sq_norm = (np.linalg.norm(etaj))**2

    e1 = np.identity(n) - (np.dot(etaj, etaj.T))/sq_norm
    a = np.dot(e1, y)

    b = etaj/sq_norm

    return a, b


def intersect(interval1, interval2):
    l1 = interval1[0]
    r1 = interval1[1]
    l2 = interval2[0]
    r2 = interval2[1]

    if max(l1, l2) > min(r1, r2):
        return None
    else:
        return [max(l1, l2), min(r1, r2)]


def construct_piecewise_quadratic(a_val, b_val, X_val, list_zk, list_active_set, list_etaAkz, list_bhAz):
    list_piecewise_quadratic = []

    for i in range(len(list_zk) - 1):
        Ak = list_active_set[i]
        lenAk = len(Ak)
        etaAk = np.array(list_etaAkz[i]).reshape((lenAk, 1))
        bhAk = np.array(list_bhAz[i]).reshape((lenAk, 1))
        zk = list_zk[i]
        XA_zk = X_val[:, Ak]

        c = np.dot(XA_zk, (bhAk - etaAk*zk))
        d = np.dot(XA_zk, etaAk)

        o = (a_val - c).flatten()
        p = (b_val - d).flatten()

        quadratic = 0
        linear = 0
        constant = 0

        for j in range(X_val.shape[0]):
            constant = constant + o[j]**2
            linear = linear + 2*o[j]*p[j]
            quadratic = quadratic + p[j]**2

        quadratic = check_zero(quadratic / 2)
        linear = check_zero(linear / 2)
        constant = check_zero(constant / 2)

        list_piecewise_quadratic.append([quadratic, linear, constant])

    return list_piecewise_quadratic


def solve_quadratic(funct_1, funct_2):
    a = funct_1[0] - funct_2[0]
    b = funct_1[1] - funct_2[1]
    c = funct_1[2] - funct_2[2]

    if a == 0:
        if b == 0:
            if c <= 0:
                return [[np.NINF, np.Inf]]
            else:
                return None
        else:
            if b < 0:
                return [[-c/b, np.Inf]]
            else:
                return [[np.NINF, -c/b]]
    else:
        delta = check_zero(b**2 - 4*a*c)
        if delta < 0:
            if a > 0:
                return None
            else:
                return [[np.NINF, np.Inf]]
        elif delta == 0:
            if a > 0:
                return [[np.NINF, -b/(2*a)]]
            else:
                return [[-b/(2*a), np.Inf]]
        else:
            x_1 = (- b - np.sqrt(delta)) / (2 * a)
            x_2 = (- b + np.sqrt(delta)) / (2 * a)

            if a > 0:
                return [[min(x_1, x_2), max(x_1, x_2)]]
            else:
                return [[np.NINF, min(x_1, x_2)], [max(x_1, x_2), np.Inf]]


def find_min_interval(list_current_funct):
    list_one_interval = []
    list_two_interval = []

    for i in range(1, len(list_current_funct)):
        return_int = solve_quadratic(list_current_funct[0], list_current_funct[i])
        if return_int is None:
            return None

        if len(return_int) == 1:
            list_one_interval.append(return_int[0])
        else:
            list_two_interval.append(return_int)

    final_one_interval = None

    for each_interval in list_one_interval:
        if final_one_interval is None:
            final_one_interval = each_interval
        else:
            final_one_interval = intersect(final_one_interval, each_interval)

    if len(list_one_interval) == 0:
        final_one_interval = [np.NINF, np.Inf]

    if final_one_interval is None:
        return None

    list_current_interval = [final_one_interval]

    for each_two_interval in list_two_interval:
        new_list_current_interval = []
        first_interval = each_two_interval[0]
        second_interval = each_two_interval[1]

        for each_iter in list_current_interval:
            temp_iter = intersect(each_iter, first_interval)

            if temp_iter is not None:
                new_list_current_interval.append(temp_iter)

        for each_iter in list_current_interval:
            temp_iter = intersect(each_iter, second_interval)

            if temp_iter is not None:
                new_list_current_interval.append(temp_iter)

        list_current_interval = new_list_current_interval

    if len(list_current_interval) == 0:
        return None

    z_interval = sorted(list_current_interval)

    new_z_interval = []

    for each_inter in z_interval:
        if len(new_z_interval) == 0:
            new_z_interval.append(each_inter)
        else:
            last_inter = new_z_interval[-1]
            if each_inter[0] <= last_inter[1]:
                new_z_interval[-1][1] = each_inter[1]
            else:
                new_z_interval.append(each_inter)

    return new_z_interval


def construct_z_interval_cv(set_piecewise_funct, set_list_zk):
    z_interval = []

    new_list_zk = []
    for each_list_zk in set_list_zk:
        new_list_zk = new_list_zk + each_list_zk

    new_list_zk = sorted(list(set(new_list_zk)))

    for i in range(1, len(new_list_zk)):
        current_z = new_list_zk[i]

        list_current_funct = []

        for j in range(len(set_list_zk)):
            list_zk_current = set_list_zk[j]

            for k in range(1, len(list_zk_current)):
                if list_zk_current[k-1] <= current_z <= list_zk_current[k]:
                    current_funct = set_piecewise_funct[j][k-1]
                    list_current_funct.append(current_funct)
                    break

        return_min_interval = find_min_interval(list_current_funct)

        if return_min_interval is None:
            continue

        for each_interval in return_min_interval:
            intersect_int = intersect(each_interval, [new_list_zk[i-1], current_z])
            if intersect_int is not None:
                z_interval.append(intersect_int)

    new_z_interval= []
    for each_interval in z_interval:
        if len(new_z_interval) == 0:
            new_z_interval.append(each_interval)
        else:
            if each_interval[0] <= new_z_interval[-1][1]:
                new_z_interval[-1][1] = each_interval[1]
            else:
                new_z_interval.append(each_interval)

    z_interval = new_z_interval

    return z_interval


def construct_m_z_interval(A, list_active_set, list_zk):
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
    return z_interval


def construct_z_interval(m_z_interval, h_z_interval):
    z_interval = []
    for each_inter_1 in m_z_interval:
        for each_inter_2 in h_z_interval:
            temp_inter = intersect(each_inter_1, each_inter_2)
            if temp_inter is not None:
                z_interval.append(temp_inter)

    z_interval = sorted(z_interval)

    new_z_interval = []

    for each_inter in z_interval:
        if len(new_z_interval) == 0:
            new_z_interval.append(each_inter)
        else:
            last_inter = new_z_interval[-1]
            if each_inter[0] <= last_inter[1]:
                new_z_interval[-1][1] = each_inter[1]
            else:
                new_z_interval.append(each_inter)

    return new_z_interval


def run():
    n = 100
    p = 5
    list_lamda = [2 ** -10, 2 ** -9, 2 ** -8, 2 ** -7, 2 ** -6, 2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 2 ** 0,
                  2 ** 1, 2 ** 2, 2 ** 3, 2 ** 4, 2 ** 5, 2 ** 6, 2 ** 7, 2 ** 8, 2 ** 9, 2 ** 10]

    beta_vec = [1, 1, 0, 0, 0]

    cov = np.identity(n)

    threshold = 20

    X, y, true_y = gen_data.generate(n, p, beta_vec)

    cutoff = int(4 * n / 5)

    X_train = X[:cutoff, :]
    y_train = y[:cutoff]

    X_val = X[cutoff:n, :]
    y_val = y[cutoff:n]

    min_cv_error = np.Inf
    lamda = None
    lamda_idx = None

    for i in range(len(list_lamda)):
        each_lamda = list_lamda[i]

        clf_lamda = linear_model.Lasso(alpha=each_lamda, fit_intercept=False, normalize=False)
        clf_lamda.fit(X_train, y_train)
        bh_lamda = clf_lamda.coef_
        bh_lamda = bh_lamda.reshape((len(bh_lamda), 1))
        temp_cv_error = 0.5*sum((y_val - (np.dot(X_val, bh_lamda)).flatten())**2)
        if temp_cv_error < min_cv_error:
            min_cv_error = temp_cv_error
            lamda = each_lamda
            lamda_idx = i

    clf = linear_model.Lasso(alpha=lamda, fit_intercept=False, normalize=False)
    clf.fit(X, y)
    bh = clf.coef_

    y = y.reshape((n, 1))
    true_y = true_y.reshape((n, 1))

    A, XA, Ac, XAc, bhA = util.construct_A_XA_Ac_XAc_bhA(X, bh, n, p)

    if len(A) == 0:
        return None

    for j_selected in A:

        etaj, etajTy = util.construct_test_statistic(j_selected, XA, y, A)

        a, b = compute_a_b(y, etaj, n)
        a_flatten = a.flatten()
        b_flatten = b.flatten()
        a_train = (a_flatten[:cutoff]).reshape((cutoff, 1))
        b_train = (b_flatten[:cutoff]).reshape((cutoff, 1))

        a_val = (a_flatten[cutoff:n]).reshape((n - cutoff, 1))
        b_val = (b_flatten[cutoff:n]).reshape((n - cutoff, 1))

        list_zk_min_lamda, list_bhz_min_lamda, list_active_set_min_lamda, list_etaAkz_min_lamda, list_bhAz_min_lamda = \
            parametric_lasso.run_parametric_lasso_cv(X_train, list_lamda[lamda_idx], X_train.shape[0], p, threshold, a_train, b_train)

        piecewise_quadratic_min_lamda = construct_piecewise_quadratic(a_val, b_val, X_val, list_zk_min_lamda,
                                                                      list_active_set_min_lamda, list_etaAkz_min_lamda,
                                                                      list_bhAz_min_lamda)

        set_piecewise_funct = [piecewise_quadratic_min_lamda]
        set_list_zk = [list_zk_min_lamda]

        for i in range(len(list_lamda)):
            if i == lamda_idx:
                continue

            list_zk_i, list_bhz_i, list_active_set_i, list_etaAkz_i, list_bhAz_i = \
                parametric_lasso.run_parametric_lasso_cv(X_train, list_lamda[i], X_train.shape[0], p, threshold, a_train, b_train)

            piecewise_quadratic_i = construct_piecewise_quadratic(a_val, b_val, X_val, list_zk_i,
                                                                  list_active_set_i, list_etaAkz_i, list_bhAz_i)

            set_piecewise_funct.append(piecewise_quadratic_i)
            set_list_zk.append(list_zk_i)

        z_interval_cv = construct_z_interval_cv(set_piecewise_funct, set_list_zk)

        list_zk, list_bhz, list_active_set = parametric_lasso.run_parametric_lasso(X, y, lamda, etaj, n, p, threshold)

        z_interval_m = construct_m_z_interval(A, list_active_set, list_zk)

        z_interval = construct_z_interval(z_interval_m, z_interval_cv)

        pivot = util.pivot_with_specified_interval(z_interval, etaj, etajTy, cov, 0)

        p_value = 2 * min(1 - pivot, pivot)

        print('Feature', j_selected + 1, ' True Beta:', beta_vec[j_selected], ' p-value:', '{:.4f}'.format(p_value))
        print("==========")


if __name__ == '__main__':
    run()
