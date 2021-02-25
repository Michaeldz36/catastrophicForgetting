import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils.utils import Setup, Teacher, Student, PrepareData, \
    load_data, train_valid_loop
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter, defaultdict, deque
import pandas as pd
#TODO: unify with code from main.py

setup = Setup()
teacher1 = Teacher()
teacher2 = Teacher()

### Hyperparameters
lr = 1e-3
epochs1 = 1000
epochs2 = 0
sgm_e = setup.sgm_e
sgm_w1 = setup.sgm_w * 1
sgm_w2 = setup.sgm_w * 2
sgm_w0=1e-3
sparsity=1 ### to ensure w(0) = 0

d=1

N = 300

P1 = 300
P2 = 300

def main(alpha, save_epochs):
    P1 = int(alpha * N)
    P2 = int(alpha * N)

    X1, Y1, w_bar1 = teacher1.build_teacher(N, P1, sgm_w1, sgm_e)
    X2, Y2, w_bar2 = teacher1.build_teacher(N, P2, sgm_w2, sgm_e)
    X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size = 0.33, random_state = 42)
    X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size = 0.33, random_state = 42)

    model = Student(n_features=N, sgm_w0=sgm_w0, sparsity=sparsity, depth = d)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # datasets for training the student network
    train_ds1 = PrepareData(X1_train, y=Y1_train, scale_X=True)
    train_ds2 = PrepareData(X2_train, y=Y2_train, scale_X=True)
    # datasets for validation errors
    valid_ds1 = PrepareData(X1_test, y=Y1_test, scale_X=True)
    valid_ds2 = PrepareData(X2_test, y=Y2_test, scale_X=True)
    # datasets for cross generalization error, TODO: not used in this simulation, make the main loop skip them
    cross_gen_ds1 = train_ds2
    cross_gen_ds2 = train_ds1

    print("Lesson 1/2")
    data_loaders1, data_lengths1 = load_data(train_ds=train_ds1, valid_ds=valid_ds1,
                                             generalize_ds=cross_gen_ds1, batch_size=X1_train.shape[0])

    history1 = train_valid_loop(data_loaders=data_loaders1,
                     data_lengths=data_lengths1,
                     n_epochs=epochs1,
                     optimizer=optimizer,
                     model=model,
                     criterion=criterion,
                     e_print=50
                    )
    print('Lesson 2/2')
    data_loaders2, data_lengths2 = load_data(train_ds=train_ds2, valid_ds=valid_ds2,
                                             generalize_ds=cross_gen_ds2, batch_size=X2_train.shape[0])

    history2 = train_valid_loop(data_loaders=data_loaders2,
                               data_lengths=data_lengths2,
                               n_epochs=epochs2,
                               optimizer=optimizer,
                               model=model,
                               criterion=criterion,
                               e_print=50
                              )

    history=np.array(history1['E_valid']+history2['E_valid'])
    Egs = {epoch: history[epoch-1] for epoch in save_epochs}
    return Egs


def simulate(alpha, n_runs, save_epochs):
    realisations = []
    for r in range(n_runs):
        print('-' * 30)
        print("Realisation {}/{}".format(r, n_runs))
        print('-' * 30)
        history = main(alpha, save_epochs=save_epochs)
        realisations.append(history)

    c = Counter()  # sums values in lists for each computed error
    v = Counter()
    for r in realisations:
        c.update(r)
        v.update({k: v ** 2 for k, v in r.items()})
    # averaging over teacher realisations
    errors = {k: v / n_runs for k, v in dict(c).items()}
    variances = {k: v/n_runs - errors[k] for k, v in dict(v).items()}
    return errors, variances

def make_data(n_runs, resolution=10, save_epochs=[epochs1, epochs1+epochs2]):
    egs_vs_alpha=defaultdict(deque)
    variances_vs_alpha=defaultdict(deque)
    alphas=np.linspace(2.5/(resolution+1), 2.5, resolution)
    for alpha in alphas:  # TODO: crashes for small N,P
        print('-'*50)
        print("Calculating for alpha = {}".format(round(alpha,2)))
        print('-'*50)

        errors, variances = simulate(alpha=alpha, n_runs=n_runs, save_epochs=save_epochs)
        for k, v in errors.items():
            egs_vs_alpha[k].append(v)
            variances_vs_alpha[k].append(variances[k])
        print('-' * 50)
        print("Finished {} %".format(round((alpha-1)/(1.5)*100,2)))
        print('-' * 50)
    errors=pd.DataFrame(egs_vs_alpha,index=alphas)
    variances=pd.DataFrame(variances_vs_alpha)
    return errors, variances

def make_plot(errors, variances=None):
    errors.plot(figsize=(8, 5), yerr=variances)
    plt.grid(True)
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$E_g(\alpha)$")
    plt.title("(MSE averaged over {} realisations)".format(n_runs))
    plt.legend(title="Times:")
    textstr = '\n'.join((
        r'$N=%.2f$' % (N,),
        r'$P_{1}=%.2f$' % (P1,),
        r'$P_{2}=%.2f$' % (P2,),
        r'$\sigma_{w_1}=%.2f$' % (sgm_w1,),
        r'$\sigma_{w_2}=%.2f$' % (sgm_w1,),
        r'$\sigma_{\epsilon}=%.2f$' % (sgm_e,),
        r'$\eta=%.2f$' % (lr,)
    ))
    plt.gcf().text(0.91, 0.12, textstr, fontsize=5)
    plt.show()

if __name__ == '__main__':
    n_runs = 1 #used for averaging over realisations
    resolution = 25 #for how many different alphas in range [0+\eps, 2.5] simulation is performed
    errors, variances = make_data(n_runs, resolution,
                       save_epochs=[epochs1//200, epochs1//50, epochs1//20, epochs1//10, epochs1, epochs1+epochs2//2, epochs1+epochs2])
    make_plot(errors=errors, variances=None)