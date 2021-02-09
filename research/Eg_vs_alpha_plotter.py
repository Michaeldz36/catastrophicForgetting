import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import Setup, Teacher, Student, PrepareData, \
    load_data, train_valid_loop
from sklearn.model_selection import train_test_split
import numpy as np


setup = Setup()
teacher1 = Teacher()
teacher2 = Teacher()

### Hyperparameters
batch_size=setup.P
learning_rate = 1e-2
epochs1 = 100
epochs2 = 100
sgm_e = setup.sgm_e
sgm_w1 = setup.sgm_w * 1
sgm_w2 = setup.sgm_w * 2

N = setup.N * 1

def main(alpha):
    P1 = int(alpha * N)
    P2 = int(alpha * N)

    X1, Y1 = teacher1.build_teacher(N, P1, sgm_w1, sgm_e)
    X2, Y2 = teacher1.build_teacher(N, P2, sgm_w2, sgm_e)
    X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size = 0.33, random_state = 42)
    X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size = 0.33, random_state = 42)

    model = Student(n_features=N, sgm_e=sgm_e)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
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

    train_valid_loop(data_loaders=data_loaders1,
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

    history = train_valid_loop(data_loaders=data_loaders2,
                               data_lengths=data_lengths2,
                               n_epochs=epochs2,
                               optimizer=optimizer,
                               model=model,
                               criterion=criterion,
                               e_print=50
                              )

    return history['E_train'][-1]

def make_data(resolution=10):
    eg_vs_alpha = []
    for alpha in np.linspace(1, 2.5, resolution):  # TODO: crashes for small N,P
        Eg = main(alpha)
        eg_vs_alpha.append(Eg)
    return eg_vs_alpha


if __name__ == '__main__':
    n_runs = 1
    resolution = 30
    total = np.empty(resolution)
    for r in range(n_runs):
        print("Run {}/{}".format(r, n_runs))
        total += np.array(make_data(resolution))
    average=total/n_runs
    # pd.DataFrame(average).plot(figsize=(8, 5))
    plt.plot(np.linspace(1, 2.5, resolution), average)
    plt.grid(True)
    plt.xlabel("alpha")
    plt.ylabel("Mean Squared Error")
    plt.title("(MSE averaged over {} realisations)".format(n_runs))
    plt.show()