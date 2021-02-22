import numpy as np
import matplotlib.pyplot as plt
from utils.utils import Setup
from utils.random import MP_evals

setup = Setup()

### Generating random eigenvalues
N=300


P1=300
eVals1 = MP_evals(P1,N)
assert len(eVals1)==N

P2=500
eVals2 = MP_evals(P2,N)
assert len(eVals2)==N

### Hyperparameters
tau = 1
epochs1 = 50
epochs2 = 50
sgm_e = setup.sgm_e
sgm_w1 = setup.sgm_w * 1
sgm_w2 = setup.sgm_w * 2
sgm_w0 = 0
weights_correlation = 0

def E_g11(t):
    A0 = (sgm_w1**2 + sgm_w0**2)
    A = lambda t: A0*np.exp(-2 * eVals1* t/tau)
    B0= 1/eVals1 * sgm_e**2
    B = lambda t: B0*(1 - np.exp(-eVals1 * t/tau))**2
    return 1 / N * np.sum(A(t) + B(t)) + sgm_e**2


def E_g22(t):
    A0 = (sgm_w2**2 + sgm_w0**2 * np.exp(-2*eVals1*epochs1/tau)+
         (sgm_w1**2 + 1 / eVals1 * sgm_e**2)*(1-np.exp(-eVals1*epochs1/tau)**2))
    A = lambda t: A0 * np.exp(-2*eVals2*(t-epochs1)/tau)
    B = lambda t: 1/ eVals2 * sgm_e**2 *(1 - np.exp(-eVals2*(t-epochs1)/tau))**2
    return 1 / N * np.sum(A(t) + B(t)) + sgm_e**2


def E_g12(t):
    A = sgm_e ** 2 + sgm_w1 ** 2
    B = lambda t: sgm_w0 ** 2 * 1 / N * np.sum(np.exp(-2 * eVals1 * epochs1 / tau) * np.exp(-2 * eVals2 * (t - epochs1) / tau))
    C = lambda t: 1 / N * np.sum((sgm_w1 ** 2 + 1 / eVals1 * sgm_e ** 2) * (1 - np.exp(-eVals1 * epochs1 / tau)) ** 2 * np.exp(
        -2 * eVals2 * (t - epochs1) / tau))
    D = lambda t:  1 / N * np.sum((sgm_w2 ** 2 + 1 / eVals2 * sgm_e ** 2) * (1 - np.exp(-eVals2 * (t - epochs1) / tau)) ** 2)
    E = lambda t: - 2 / N**2 * np.sum((1-np.exp(-eVals2*(t-epochs1)/tau)) * weights_correlation)
    return A+B(t)+C(t)+D(t)+E(t)


def make_plot(epochs1=50, epochs2=50, cross_gen=False):
    timesteps = np.linspace(0, epochs1, 150)
    e_g11 = np.array([E_g11(_t) for _t in timesteps])
    plt.plot(timesteps, e_g11 / sgm_w1 ** 2, label="Generalization Error (11")

    if epochs2:
        timesteps2 = np.linspace(epochs1, epochs1 + epochs2, 150)
        e_g22 = np.array([E_g22(_t) for _t in timesteps2])
        plt.plot(timesteps2, e_g22 / sgm_w2 ** 2, label="Generalization Error (22)")
        if cross_gen:
            e_g12 = np.array([E_g12(_t) for _t in timesteps2])
            plt.plot(timesteps2, e_g12 / sgm_w2 ** 2, label="Generalization Error (12)")

    plt.xlabel("Time")
    plt.ylabel("Generalization Error")
    textstr = '\n'.join((
        r'$N=%.2f$' % (N,),
        r'$P_{1}=%.2f$' % (P1,),
        r'$P_{2}=%.2f$' % (P2,),
        r'$\sigma_{w_1}=%.2f$' % (sgm_w1,),
        r'$\sigma_{w_2}=%.2f$' % (sgm_w1,),
        r'$\sigma_{\epsilon}=%.2f$' % (sgm_e,),
    ))
    plt.gcf().text(0.91, 0.12, textstr, fontsize=5)
    plt.grid(True)
    # plt.ylim([0,2])
    plt.legend()
    plt.show()
    print("Done!")

if __name__ == '__main__':
    make_plot(epochs1=50, epochs2=70, cross_gen=True)





