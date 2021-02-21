import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from utils.random import MP_evals
"""To test if scaling of MP eigenvalues is right look if empirically generated PDF using KDE 
is comparable to analytical one"""


P = 10000
N = 1000


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

# Generating a random matrix
eVals = MP_evals(P,N, diagonal=True)
pdf_MP = mpPDF(alpha=P/N, var=1, pts=1000)
pdf_empirical= fitKDE(np.diag(eVals), bWidth=0.01)

if __name__ == '__main__':
    pdf_MP.plot(title="Marchenko-Pastur Theorem", label="Marchenko-Pastur", color="deepskyblue")
    pdf_empirical.plot(label="Empirical Value", color="red")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$\rho(\lambda)$")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.show()