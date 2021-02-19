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
epochs1 = 250
epochs2 = 250
sgm_e = setup.sgm_e
sgm_w1 = setup.sgm_w * 1
sgm_w2 = setup.sgm_w * 2

N = 20

P1 = 20
P2 = 20

def main(alpha, save_epochs=[epochs1, epochs2]):
    P1 = int(alpha * N)
    P2 = int(alpha * N)

    X1, Y1 = teacher1.build_teacher(N, P1, sgm_w1, sgm_e)
    X2, Y2 = teacher1.build_teacher(N, P2, sgm_w2, sgm_e)
    X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size = 0.33, random_state = 42)
    X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size = 0.33, random_state = 42)

    model = Student(n_features=N, sgm_e=sgm_e)
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
    Egs = {epoch: history[epoch] for epoch in save_epochs}
    return Egs


def simulate(alpha, n_runs, save_epochs):
    realisations = []
    for r in range(n_runs):
        print("Realisation {}/{}".format(r, n_runs))
        history = main(alpha, save_epochs=save_epochs)
        realisations.append(history)
    c = Counter()  # sums values in lists for each computed error
    for r in realisations:
        c.update(r)
    # averaging over teacher realisations
    egs_averaged = {k: v / n_runs for k, v in dict(c).items()}
    return egs_averaged

def make_data(n_runs, resolution=10, save_epochs=[epochs1, epochs2]):
    egs_vs_alpha=defaultdict(deque)
    for alpha in np.linspace(1, 2.5, resolution):  # TODO: crashes for small N,P
        print('-'*42)
        print("Calculating for alpha = {}, finished {} %".format(round(alpha,2), round((alpha-1)/(1.5)*100,2)))
        print('-'*42)

        averaged_egs=simulate(alpha, n_runs, save_epochs=save_epochs)
        for k, v in averaged_egs.items():
            egs_vs_alpha[k].append(v)
    errors=pd.DataFrame(egs_vs_alpha)
    return errors

def make_plot(errors, resolution):
    errors.plot(figsize=(8, 5))
    # plt.plot(np.linspace(1, 2.5, resolution), average)
    plt.grid(True)
    plt.xlabel("alpha")
    plt.ylabel("Mean Squared Error")
    plt.title("(MSE averaged over {} realisations)".format(n_runs))
    plt.show()

if __name__ == '__main__':
    n_runs = 10  #used for averaging over realisations
    resolution = 20 #for how many different alphas in range [1, 2.5] simulation is performed
    errors = make_data(n_runs, resolution,
                       save_epochs=[int(epochs1/2), epochs1, int(epochs1+epochs2/2), epochs2])
    make_plot(errors, resolution)