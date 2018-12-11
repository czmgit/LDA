from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np


def test_beta_distribution():
    fig, ax = plt.subplots(1, 1)
    a, b = 10, 30
    # Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).
    mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')
    print(mean)
    print(var)
    print(skew)
    print(kurt)
    print(beta.pdf(0.333, a, b))
    x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)
    ax.plot(x, beta.pdf(x, a, b), 'r-', lw=5, alpha=0.6, label='beta pdf')
    rv = beta(a, b)
    ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
    vals = beta.ppf([0.001, 0.5, 0.999], a, b)
    np.allclose([0.001, 0.5, 0.999], beta.cdf(vals, a, b))
    r = beta.rvs(a, b, size=1000)
    ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
    ax.legend(loc='best', frameon=False)
    plt.show()


if __name__ == '__main__':
    test_beta_distribution()