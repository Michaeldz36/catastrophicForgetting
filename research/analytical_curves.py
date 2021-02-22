import numpy as np
import matplotlib.pyplot as plt
from utils.utils import Setup
from utils.random import MP_evals


class AnalyticalSolution():
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
        self.eVals2 = MP_evals(self.P2,self.N)
        assert len(self.eVals2)==self.N



    def E_g11(self,t):
        A0 = (self.sgm_w1**2 + self.sgm_w0**2)
        A = lambda t: A0*np.exp(-2 * self.eVals1* t/self.tau)
        B0= 1/self.eVals1 * self.sgm_e**2
        B = lambda t: B0*(1 - np.exp(-self.eVals1 * t/self.tau))**2
        return 1 / self.N * np.sum(A(t) + B(t)) + self.sgm_e**2


    def E_g22(self,t):
        A0 = (self.sgm_w2**2 + self.sgm_w0**2 * np.exp(-2*self.eVals1*self.epochs1/self.tau)+
             (self.sgm_w1**2 + 1 / self.eVals1 * self.sgm_e**2)*(1-np.exp(-self.eVals1*self.epochs1/self.tau)**2))
        A = lambda t: A0 * np.exp(-2*self.eVals2*(t-self.epochs1)/self.tau)
        B = lambda t: 1/ self.eVals2 * self.sgm_e**2 *(1 - np.exp(-self.eVals2*(t-self.epochs1)/self.tau))**2
        return 1 / self.N * np.sum(A(t) + B(t)) + self.sgm_e**2


    def E_g12(self, t):
        A0 = np.exp(-2 * self.eVals1 * self.epochs1 / self.tau)
        A = lambda t: self.sgm_w0 ** 2 * 1 / self.N * np.sum(A0 * np.exp(-2 * self.eVals2 * (t - self.epochs1) / self.tau))
        B0 = (self.sgm_w1 ** 2 + 1 / self.eVals1 * self.sgm_e ** 2)* (1 - np.exp(-self.eVals1 * self.epochs1 / self.tau)) ** 2
        B = lambda t: 1 / self.N * np.sum(B0  * np.exp(-2 * self.eVals2 * (t - self.epochs1) / self.tau))
        C0 = (self.sgm_w2 ** 2 + 1 / self.eVals2 * self.sgm_e ** 2)
        C = lambda t:  1 / self.N * np.sum(C0 * (1 - np.exp(-self.eVals2 * (t - self.epochs1) / self.tau)) ** 2)
        D = lambda t: - 2 / self.N**2 * np.sum((1-np.exp(-self.eVals2*(t-self.epochs1)/self.tau)) * self.weights_correlation)
        return A(t)+B(t)+C(t)+D(t) + self.sgm_e ** 2 + self.sgm_w1 ** 2

    def curves(self, timesteps1, timesteps2):
        e_g11 = np.array([AnalyticalSolution.E_g11(self,_t) for _t in timesteps1])
        e_g22 = np.array([AnalyticalSolution.E_g22(self,_t) for _t in timesteps2])
        e_g12 = np.array([AnalyticalSolution.E_g12(self,_t) for _t in timesteps2])
        return e_g11, e_g22, e_g12




def make_plot(e_g11, e_g22, e_g12, timesteps1, timesteps2):
    plt.plot(timesteps1, e_g11, label="Generalization Error (11")
    plt.plot(timesteps2, e_g22, label="Generalization Error (22)")
    plt.plot(timesteps2, e_g12, label="Generalization Error (12)")
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
    # plt.ylim([0,2])
    plt.legend()
    plt.show()
    print("Done!")

if __name__ == '__main__':
    setup = Setup()
    curves = AnalyticalSolution(N=300, P1=300, P2=500, tau=1,
                                epochs1=50, epochs2=50, sgm_e=setup.sgm_e,
                                sgm_w1=setup.sgm_w, sgm_w2=setup.sgm_w * 2,
                                sgm_w0=setup.sgm_w0, weights_correlation=0)
    timesteps1 = np.linspace(0, curves.epochs1, 150)
    timesteps2 = np.linspace(curves.epochs1, curves.epochs1 + curves.epochs2,150)
    e_g11, e_g22, e_g12 = curves.curves(timesteps1, timesteps2)
    plt.title("Analytical solutions to 2 linear teacher scenario")
    make_plot(e_g11/curves.sgm_w1**2, e_g22/curves.sgm_w2**2, e_g12/curves.sgm_w2**2, timesteps1, timesteps2)






