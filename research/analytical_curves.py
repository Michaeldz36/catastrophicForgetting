import numpy as np
import matplotlib.pyplot as plt
from utils.utils import Setup
from utils.random import MP_evals

setup = Setup()

### Generating random eigenvalues
N=100


P1=1000
eVals1 = MP_evals(P1,N)
assert len(eVals1)==N

P2=1000
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


E_g11 = lambda t: 1 / N * np.sum( (sgm_w1**2 + sgm_w0**2)*np.exp(-2 * eVals1* t/tau)
                                  +1/eVals1 * sgm_e**2 * (1 - np.exp(-eVals1 * t/tau))**2 ) + sgm_e**2


E_g22 = lambda t: 1 / N * \
        np.sum((sgm_w2**2 + sgm_w0**2 * np.exp(-2*eVals1*epochs1/tau)+
         (sgm_w1**2 + 1 / eVals1 * sgm_e**2)*(1-np.exp(-eVals1*epochs1/tau)**2))*np.exp(-2*eVals2*(t-epochs1)/tau) +
         1/ eVals2 * sgm_e**2 *(1 - np.exp(-eVals2*(t-epochs1)/tau))**2) + sgm_e**2

if __name__ == '__main__':
    timesteps = np.linspace(0,epochs1,150)
    e_g11 = np.array([E_g11(_t) for _t in timesteps])
    plt.plot(timesteps, e_g11/sgm_w1**2, label="Generalization Error (11")
    plt.xlabel("Time")
    plt.ylabel("Generalization Error")
    plt.grid(True)
    plt.legend()
    plt.show()
    print("Done!")

    timesteps2 = np.linspace(epochs1, epochs1+epochs2, 150)
    e_g22 = np.array([E_g22(_t) for _t in timesteps2])
    plt.plot(timesteps2, e_g22 / sgm_w2 ** 2, label="Generalization Error (22)")
    plt.xlabel("Time")
    plt.ylabel("Generalization Error")
    plt.grid(True)
    plt.legend()
    plt.show()
    print("Done!")
