import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
from utils.utils import Setup
from utils.random import MP_evals


class AnalyticalSolution: #TODO: make it work with main's syllabus
    def __init__(self, N, P1, P2, tau, epochs1, epochs2, sgm_e, sgm_w1, sgm_w2, sgm_w0, weights_correlation):
        ### Hyperparameters
        self.tau = tau
        self.epochs1 = epochs1
        self.epochs2 = epochs2
        self.sgm_e = sgm_e
        self.sgm_w1 = sgm_w1
        self.sgm_w2 = sgm_w2
        self.sgm_w0 = sgm_w0
        self.weights_correlation = weights_correlation

        ### Generating random eigenvalues
        self.N=N
        self.P1=P1
        self.eVals1 = MP_evals(self.P1,self.N)
        assert len(self.eVals1)==self.N

        self.P2=P2
        self.eVals2 = MP_evals(self.P2,self.N)[::1]
        assert len(self.eVals2)==self.N



    def E_g11(self,t):
        A0 = (self.sgm_w1**2 + self.sgm_w0**2)
        A = lambda t: A0*np.exp(-2 * self.eVals1* t/self.tau)
        B0= 1/self.eVals1 * self.sgm_e**2
        B = lambda t: B0*(1 - np.exp(-self.eVals1 * t/self.tau))**2
        eg11= 1 / self.N * np.sum(A(t) + B(t)) + self.sgm_e**2
        return eg11


    def E_g22(self,t):
        A0 = (self.sgm_w2**2 + self.sgm_w0**2 * np.exp(-2*self.eVals1*self.epochs1/self.tau)+
             (self.sgm_w1**2 + 1 / self.eVals1 * self.sgm_e**2)*(1-np.exp(-self.eVals1*self.epochs1/self.tau)**2))
        A = lambda t: A0 * np.exp(-2*self.eVals2*(t-self.epochs1)/self.tau)
        B = lambda t: 1/ self.eVals2 * self.sgm_e**2 *(1 - np.exp(-self.eVals2*(t-self.epochs1)/self.tau))**2
        eg22 =1 / self.N * np.sum(A(t) + B(t)) + self.sgm_e**2 #  shouldnt there be + self.sgm_w1**2 ??
        return eg22

    def E_g12(self, t):
        A0 = np.exp(-2 * self.eVals1 * self.epochs1 / self.tau)
        A = lambda t: self.sgm_w0 ** 2 * 1 / self.N * np.sum(A0 * np.exp(-2 * self.eVals2 * (t - self.epochs1) / self.tau))
        B0 = (self.sgm_w1 ** 2 + 1 / self.eVals1 * self.sgm_e ** 2)* (1 - np.exp(-self.eVals1 * self.epochs1 / self.tau)) ** 2
        B = lambda t: 1 / self.N * np.sum(B0  * np.exp(-2 * self.eVals2 * (t - self.epochs1) / self.tau))
        C0 = (self.sgm_w2 ** 2 + 1 / self.eVals2 * self.sgm_e ** 2)
        C = lambda t:  1 / self.N * np.sum(C0 * (1 - np.exp(-self.eVals2 * (t - self.epochs1) / self.tau)) ** 2)
        D = lambda t: - 2 / self.N**2 * np.sum((1-np.exp(-self.eVals2*(t-self.epochs1)/self.tau)) * self.weights_correlation)
        eg12= A(t)+B(t)+C(t)+D(t) + self.sgm_e ** 2 + self.sgm_w1 ** 2
        return eg12

    def curves(self, timesteps1, timesteps2):
        e_g11 = np.array([AnalyticalSolution.E_g11(self,_t) for _t in timesteps1])
        e_g22 = np.array([AnalyticalSolution.E_g22(self,_t) for _t in timesteps2])
        e_g12 = np.array([AnalyticalSolution.E_g12(self,_t) for _t in timesteps2])
        return e_g11, e_g22, e_g12




def make_plot(e_g11, e_g22, e_g12, timesteps1, timesteps2):
    plt.plot(timesteps1, e_g11, label="Generalization Error (11)", color='orange')
    plt.plot(timesteps2, e_g22, label="Generalization Error (22)", color='orange')
    plt.plot(timesteps2, e_g12, label="Generalization Error (12)", color='orange', linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Generalization Error")
    textstr = '\n'.join((
        r'$N=%.2f$' % (curves.N,),
        r'$P_{1}=%.2f$' % (curves.P1,),
        r'$P_{2}=%.2f$' % (curves.P2,),
        r'$\sigma_{w_1}=%.2f$' % (curves.sgm_w1,),
        r'$\sigma_{w_2}=%.2f$' % (curves.sgm_w2,),
        r'$\sigma_{\epsilon}=%.2f$' % (curves.sgm_e,),
    ))
    plt.gcf().text(0.91, 0.12, textstr, fontsize=5)
    plt.grid(True)
    plt.ylim([0,1.75])
    plt.legend()
    plt.axhline(y=0.2**2 +1**2+0**2, color='r', linestyle='--',  linewidth=0.5, alpha=0.5)
    plt.show()
    print("Done!")

if __name__ == '__main__':
    setup = Setup()
    for tau in [1/2, 1,  1.5, 2]:
        curves = AnalyticalSolution(N=100, P1=100, P2=100, tau=tau,
                                    epochs1=1000, epochs2=1000, sgm_e=setup.sgm_e,
                                    sgm_w1=setup.sgm_w, sgm_w2=setup.sgm_w * 2,
                                    sgm_w0=setup.sgm_w0, weights_correlation=0)
        timesteps1 = np.linspace(0, curves.epochs1, 150)
        timesteps2 = np.linspace(curves.epochs1, curves.epochs1 + curves.epochs2,150)
        e_g11, e_g22, e_g12 = curves.curves(timesteps1, timesteps2)
        plt.title("Analytical solutions to 2 linear teacher scenario WITH {}".format(tau))
        make_plot(e_g11, e_g22, e_g12, timesteps1, timesteps2)






