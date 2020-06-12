# Parametric Programming Approach for More Powerful and General Lasso Selective Inference

This package implements a parametric programming approach for more powerful and general Lasso Selective Inference (SI). The main idea is to compute the continuum path of Lasso solutions in the direction of the selected test statistic, and identify the subset of the data space corresponding to the feature selection event by following the solution path. The proposed parametric programming-based method not only avoids all the drawback of current Lasso SI methods but also improves the performance and practicality of SI for Lasso in various respects.

See the paper <https://arxiv.org/abs/2004.09749> for more details.

## Installation & Requirements

This package has the following requirements:

- [numpy](http://numpy.org)
- [mpmath](http://mpmath.org/)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org)
- [statsmodels](https://www.statsmodels.org/)

We recommend to install or update anaconda to the latest version and use Python 3
(We used Python 3.6.4).

## Reproducibility

**NOTE**: In the code, we use the following optimization objective for Lasso:

(1 / (2 * n_samples)) * ||y - X beta||^2_2 + lambda * ||beta||_1

All the figure results are saved in folder "/results" and some results are shown on console.

The following commands are run from the terminal.

- Checking the uniformity of the pivot in Lasso case
```
>> python ex1_pivot_lasso.py
```

- Example for computing p-value
```
>> python ex2_p_value.py
```

- Example for computing confidence interval
```
>> python ex3_confidence_interval.py
```

- Example for computing p-value when additional consider selection event of tuning hyperparameter
```
>> python ex4_test_with_cross_validation_event.py
```

- Checking the uniformity of the pivot in Elastic Net case
```
>> python ex5_pivot_elastic_net.py
```

