import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import Setup, Teacher, Student, PrepareData, \
    load_data, train_valid_loop
from sklearn.model_selection import train_test_split



setup = Setup()
teacher1 = Teacher()
teacher2 = Teacher()

### Hyperparameters
batch_size=setup.N
learning_rate = 1e-2
epochs1 = 500
epochs2=500


def main(alpha_ratio):
    N1 = setup.N * 1  # for now N=N1=N2
    N2 = setup.N * 1  # possible TODO: make N=max(N1,N2), X=[max(X1,X2),min(X1,X2).concat(zeros)]

    P1 = int(alpha_ratio/10 * setup.N)
    P2 = int(alpha_ratio/10 * setup.N * 2)

    sgm_w1 = setup.sgm_w * 1
    sgm_w2 = setup.sgm_w * 2

    X1, Y1 = teacher1.build_teacher(N1, P1, sgm_w1, setup.sgm_e)
    X2, Y2 = teacher1.build_teacher(N2, P2, sgm_w2, setup.sgm_e)
    X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size = 0.33, random_state = 42)
    X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size = 0.33, random_state = 42)

    X_train, Y_train = np.c_[X1_train.T, X2_train.T].T, np.r_[Y1_train, Y2_train]
    X_test, Y_test = np.c_[X1_test.T, X2_test.T].T, np.r_[Y1_test, Y2_test]


    model = Student(n_features=setup.N, sgm_e=setup.sgm_e)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()


    train_ds1 = PrepareData(X1_train, y=Y1_train, scale_X=True)
    generalize_ds1 = PrepareData(X1_test, y=Y1_test, scale_X=True)
    data_loaders1, data_lengths1 = load_data(train_ds=train_ds1, valid_ds=generalize_ds1,
                             batch_size=X1_train.shape[0])


    history1 = train_valid_loop(data_loaders=data_loaders1,
                     data_lengths=data_lengths1,
                     n_epochs=epochs1,
                     optimizer=optimizer,
                     model=model,
                     criterion=criterion,
                     e_print=1
                     )

    train_ds2 = PrepareData(X2_train, y=Y2_train, scale_X=True)
    generalize_ds2 = PrepareData(X2_test, y=Y2_test, scale_X=True)
    data_loaders2, data_lengths2 = load_data(train_ds=train_ds2, valid_ds=generalize_ds2,
                                           batch_size=X2_train.shape[0])

    history2 = train_valid_loop(data_loaders=data_loaders2,
                               data_lengths=data_lengths2,
                               n_epochs=epochs2,
                               optimizer=optimizer,
                               model=model,
                               criterion=criterion,
                               e_print=1
                               )
    return history2['E_t'][-1]



if __name__ == '__main__':
    eg_vs_alpha = []
    for alpha_ratio in range(1, 25 + 1): #TODO: crashes for small N,P
        Eg=main(alpha_ratio)
        eg_vs_alpha.append(Eg)

    pd.DataFrame(eg_vs_alpha).plot(figsize=(8, 5))
    plt.grid(True)
    # plt.gca().set_ylim(0, 1)
    plt.show()
