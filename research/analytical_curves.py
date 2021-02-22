import numpy as np
import matplotlib.pyplot as plt
from utils.utils import Setup
from utils.random import MP_evals

setup = Setup()

### Generating random eigenvalues
P=300
N=200
eVals1 = MP_evals(P,N)
print(len(eVals1))


### Hyperparameters
tau = 1
epochs1 = 100
epochs2 = 0
sgm_e = setup.sgm_e
sgm_w1 = setup.sgm_w * 1
sgm_w2 = setup.sgm_w * 2
sgm_w0 = 0




E_g11 = lambda t: 1 / N * np.sum( (sgm_w1**2 + sgm_w0**2)*np.exp(-2 * eVals1* t/tau)
                                  +1/eVals1 * sgm_e**2 * (1 - np.exp(-eVals1 * t/tau))**2 ) + sgm_e**2

if __name__ == '__main__':
    timesteps = np.linspace(0,epochs1+epochs2,100)
    e_g11 = np.array([E_g11(_t) for _t in timesteps])
    plt.plot(timesteps, e_g11/sgm_w1**2)
    plt.xlabel("Time")
    plt.ylabel("Generalization Error")
    plt.show()
