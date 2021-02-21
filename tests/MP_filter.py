import numpy as np
from utils.random import randomCov, corr_from_cov, eigen_dcmp, \
    find_lambda_max, mpPDF, fitKDE
import pandas as pd
import matplotlib.pyplot as plt

SNR, N, M, alpha = .995, 1000, 100, 10

# Generate Random Matrix reperesenting Gaussian signal
signal = np.random.normal(size=(N * alpha, N))


# Covariance Matrix
cov = np.cov(signal, rowvar=0)

# Adding noise to Covariance Matrix
noised_cov = SNR * cov + (1 - SNR) * randomCov(N, M)

# Converting noised Covariancce to Correlation
corr = corr_from_cov(noised_cov)
# Getting Eigenvalues and Eigenvectors
eVals, _ = eigen_dcmp(corr, diagonal=True)
# Finding lambda_max and variance attributed to noise
lambda_max, noise_variance = find_lambda_max(np.diag(eVals), alpha, bWidth=0.01)
signal_amount = eVals.shape[0] - np.diag(eVals)[::-1].searchsorted(lambda_max)
print("# of signal eigenvalues:", signal_amount)

if __name__ == '__main__':
    pdf_MP = mpPDF(alpha=alpha, var=noise_variance, pts=1000)
    pdf_empirical = fitKDE(np.diag(eVals), bWidth=0.01)
    pdf_MP.plot(label="Marchenko-Pastur", color="deepskyblue")
    plt.hist(pd.Series(np.diag(eVals)), density="norm", bins=1000, label="Empirical Value", color="red")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$\rho(\lambda)$")
    plt.legend(loc="upper right")
    plt.title("Marchenko-Pastur Filter")
    plt.grid(True)
    plt.show()