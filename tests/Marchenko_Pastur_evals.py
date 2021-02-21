import numpy as np
import matplotlib.pyplot as plt
from utils.random import MP_evals, mpPDF, fitKDE

"""To test if scaling of MP eigenvalues is right look if empirically generated PDF using KDE 
is comparable to analytical one"""


P = 10000
N = 1000

# Generating a random matrix
eVals = MP_evals(P,N, diagonal=True)
pdf_MP = mpPDF(alpha=P / N, var=1, pts=1000)
pdf_empirical= fitKDE(np.diag(eVals), bWidth=0.01)

if __name__ == '__main__':
    pdf_MP.plot(title="Marchenko-Pastur Theorem", label="Marchenko-Pastur", color="deepskyblue")
    pdf_empirical.plot(label="Empirical Value", color="red")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$\rho(\lambda)$")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.show()