import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity


def eigen_dcmp(matrix, diagonal=False):
    eVals, eVecs = np.linalg.eigh(matrix) # works for hermitian matrix, e.g. corr matrix
    idx = eVals.argsort()[::-1]  # arguments for sorting eVal desc
    eVals, eVecs = eVals[idx], eVecs[:, idx]
    if diagonal:
        eVals = np.diagflat(eVals)
    return eVals, eVecs


def MP_evals(P, N, diagonal=False):
    # TODO: assert P>N for Wishart ensemble, maybe implement anti-wishart in the future?
    x = np.random.normal(size=(P, N))
    eVals, _ = eigen_dcmp(np.corrcoef(x, rowvar=0), diagonal)
    return eVals


def check_correlation(X1,X2):
    X1=X1.flatten()
    X2=X2.flatten()
    pearson = np.corrcoef(X1,X2)
    print("Correlation between data is:\n", pearson)
    return pearson


def randomCov(N, M):
    #TODO: implement vine method https://stats.stackexchange.com/questions/124538/how-to-generate-a-large-full-rank-random-correlation-matrix-with-some-strong-cor
    w = np.random.normal(size=(N, M))
    cov = np.dot(w, w.T) # not full rank
    cov += np.diag(np.random.uniform(size=N)) # full rank
    return cov  # NxN


def corr_from_cov(cov):
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1
    return corr


def fit_MP_rss_error(var, eVals, alpha, bWidth, pts=1000):
    pdf_MP = mpPDF(alpha=alpha, var=var, pts=pts)
    pdf_empirical = fitKDE(eVals, bWidth, x=pdf_MP.index.values)  # empirical pdf
    rss = np.sum((pdf_empirical - pdf_MP) ** 2)
    return rss


def mpPDF(alpha, var, pts):
    # alpha=P/N
    if isinstance(var, np.ndarray):
        if var.shape == (1,):
            var = var[0]
    lambda_min, lambda_max = alpha**-1 * var * (np.sqrt(alpha)-1)**2,  alpha**-1 * var * (np.sqrt(alpha)+1) ** 2
    x = np.linspace(lambda_min, lambda_max, pts)
    pdf = alpha / (2 * np.pi * var * x) * np.sqrt((lambda_max - x) * (x - lambda_min))
    pdf = pd.Series(pdf, index=x)
    return pdf


def fitKDE(obs, bWidth=.25, kernel='gaussian', x=None):
    if len(obs.shape) == 1:
        obs = obs.reshape(-1, 1)
    kde = KernelDensity(kernel=kernel, bandwidth=bWidth).fit(obs)
    if x is None:
        x = np.unique(obs).reshape(-1, 1)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    logProb = kde.score_samples(x)  # log(density)
    pdf = pd.Series(np.exp(logProb), index=x.flatten())
    return pdf


def find_lambda_max(eVal, alpha, bWidth):
    out = minimize(lambda *x: fit_MP_rss_error(*x), .5, args=(eVal, alpha, bWidth),
                   bounds=((1E-5, 1 - 1E-5),))
    if out['success']:
        var = out['x'][0]
    else:
        var = 1
    eMax = var * (1 + (1. / alpha) ** .5) ** 2
    return eMax, var