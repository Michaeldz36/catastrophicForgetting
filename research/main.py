import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import Setup, Teacher, Student, PrepareData, \
    load_data, train_valid_loop
from utils.random import check_correlation
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
from analytical_curves import AnalyticalSolution

setup = Setup()

### Hyperparameters
N = 300
P1 = 300
P2 = P1 # for now

sgm_w1 = setup.sgm_w * 1
sgm_w2 = setup.sgm_w * 2
sgm_e = setup.sgm_e

sgm_w0 = 1e-13 ### 0 in article..
sparsity=1 ### this hack enables us to initialize with 0 weights

epochs1 = 500
epochs2 = 500

lr = 1e-2
batch_size=P1
tau = 1/ lr * batch_size ### in article tau = Delta_t / lr  (maybe P???)
depth = 1


def main(N=N, P1=P1, P2=P2, epochs1=epochs1, epochs2=epochs2, sgm_e=sgm_e, sgm_w2=sgm_w2, sgm_w1=sgm_w1, lr=lr,
         d=depth):
    teacher1 = Teacher()
    teacher2 = Teacher()
    #TODO: use make_ds function from utils
    X1, Y1, w_bar1 = teacher1.build_teacher(N, 2*P1, sgm_w1, sgm_e) ### generate 2 times more example for train/test split
    X2, Y2, w_bar2 = teacher2.build_teacher(N, 2*P2, sgm_w2, sgm_e)
    #TODO: add KS-test? check how X1, X2 are correlated, make them look different
    # check_correlation(X1,X2) # checks Pearson correlation coefficient, works only for P1=P2
    X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.5, random_state=42)
    X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=0.5, random_state=42)


    model = Student(n_features=N, sgm_w0=sgm_w0, sparsity=sparsity, depth = d)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='mean')

    # datasets for training the student network
    train_ds1 = PrepareData(X1_train, y=Y1_train, scale_X=True)
    train_ds2 = PrepareData(X2_train, y=Y2_train, scale_X=True)
    # datasets for validation errors
    valid_ds1 = PrepareData(X1_test, y=Y1_test, scale_X=True)
    valid_ds2 = PrepareData(X2_test, y=Y2_test, scale_X=True)
    # datasets for cross generalization error
    cross_gen_ds1 = valid_ds2
    cross_gen_ds2 = valid_ds1

    print("Lesson 1/2:")
    print('-' * 20)
    data_loaders1, data_lengths1 = load_data(train_ds=train_ds1, valid_ds=valid_ds1,
                                             generalize_ds=cross_gen_ds1, batch_size=X1_train.shape[0])

    history1 = train_valid_loop(data_loaders=data_loaders1,
                                data_lengths=data_lengths1,
                                n_epochs=epochs1,
                                optimizer=optimizer,
                                model=model,
                                criterion=criterion,
                                e_print=50,
                                phases=['train','valid',]
                                )
    print('Lesson 2/2:')
    print('-' * 20)
    data_loaders2, data_lengths2 = load_data(train_ds=train_ds2, valid_ds=valid_ds2,
                                             generalize_ds=cross_gen_ds2, batch_size=X2_train.shape[0])

    history2 = train_valid_loop(data_loaders=data_loaders2,
                                data_lengths=data_lengths2,
                                n_epochs=epochs2,
                                optimizer=optimizer,
                                model=model,
                                criterion=criterion,
                                e_print=50,
                                )

    full_history = dict()
    for key in history1.keys():
        full_history[key] = np.array(history1[key] + history2[key])
    return full_history


def simulate(syllabus, n_runs):
    realisations = []
    for r in range(n_runs):
        print('Realisation {}/{}'.format(r, n_runs))
        history = main(*syllabus)
        realisations.append(history)
    # TODO: check (unit test), clean this clutter
    square_realisations=[]
    c = Counter() # sums values in lists for each computed error
    for r in realisations:
        c.update(r)
        square_realisations.append({k: v ** 2 for k, v in r.items()})
    # averaging over teacher realisations
    errors = pd.DataFrame(c) / n_runs

    v = Counter()
    for sr in square_realisations:
        v.update(sr)
    variances=pd.DataFrame(v)/ n_runs - errors**2
    return errors, variances


def plot_history(errors, n_runs, variances=None, analytical=False, yrange=2):
    errors.plot(figsize=(8, 5), yerr=None)
    for column in errors.columns:
        plt.fill_between(variances.index, errors[column]-variances[column], errors[column]+variances[column], alpha=0.2)
    plt.axhline(y=sgm_e, color='r', linestyle='--',  linewidth=0.5, alpha=0.5)
    plt.grid(True)
    plt.ylim([0,yrange])
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("Linear network with {} layers depth \n"
              "(MSE averaged over {} realisations)".format(depth, n_runs))
    textstr = '\n'.join((
        r'$N=%.2f$' % (N, ),
        r'$P_{1}=%.2f$' % (P1, ),
        r'$P_{2}=%.2f$' % (P2, ),
        r'$\sigma_{w_1}=%.2f$' % (sgm_w1,),
        r'$\sigma_{w_2}=%.2f$' % (sgm_w1,),
        r'$\sigma_{\epsilon}=%.2f$' % (sgm_e,),
        r'$\eta=%.2f$' % (lr,)
    ))
    plt.gcf().text(0.91, 0.12, textstr, fontsize=5)
    if analytical:
        analytical = AnalyticalSolution(N, P1, P2, tau, epochs1, epochs2, sgm_e, sgm_w1, sgm_w2, 0., 0.)
        timesteps1 = np.linspace(0, epochs1, epochs1)
        timesteps2 = np.linspace(epochs1, epochs1 + epochs2, epochs2)
        e_g11, e_g22, e_g12 = analytical.curves(timesteps1, timesteps2)
        plt.plot(timesteps1, e_g11, label='analytical E_g11', linestyle='--')
        plt.plot(timesteps2, e_g22, label='analytical E_g22', linestyle=':')
        plt.plot(timesteps2, e_g12, label='analytical E_g12', linestyle='-.')
        plt.legend()
    plt.show()


if __name__ == '__main__': #TODO: update jupyter notebook
    syllabus = [N, P1, P2, epochs1, epochs2, sgm_w1, sgm_w2, sgm_e, lr, depth]
    n_runs = 3
    errors, variances = simulate(syllabus, n_runs)
    plot_history(errors=errors, n_runs=n_runs,
                 variances=variances, analytical=True, yrange=2)

