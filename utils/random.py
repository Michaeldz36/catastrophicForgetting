import numpy as np


def eigen_dcmp(matrix, diagonal=False):
    eVals, eVecs = np.linalg.eigh(matrix) # works for hermitian matrix
    idx = eVals.argsort()[::-1]  # arguments for sorting eVal desc
    eVals, eVecs = eVals[idx], eVecs[:, idx]
    if diagonal:
        eVals = np.diagflat(eVals)
    return eVals, eVecs


def MP_evals(P, N, diagonal=False):
    x = np.random.normal(size=(P, N))
    eVals, _ = eigen_dcmp(np.corrcoef(x, rowvar=0), diagonal)
    return eVals


def check_correlation(X1,X2):
    X1=X1.flatten()
    X2=X2.flatten()
    pearson = np.corrcoef(X1,X2)
    print("Correlation between data is:\n", pearson)
    return pearson