import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import Setup, Teacher, Student, make_ds, training_loop
from sklearn.model_selection import train_test_split



setup = Setup()
teacher1 = Teacher()
teacher2 = Teacher()

### Hyperparameters
batch_size=setup.P
learning_rate = 1e-2
epochs = 500


def main():
    N1 = setup.N * 1  # for now N=N1=N2
    N2 = setup.N * 1  # possible TODO: make N=max(N1,N2), X=[max(X1,X2),min(X1,X2).concat(zeros)]

    P1 = setup.P * 1
    P2 = setup.P * 2

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

    X,Y = make_ds(X_train, Y_train, scale_X=True)
    history = training_loop(X_train=X, Y_train=Y, n_epochs=epochs,
                            optimizer=optimizer,
                            model=model,
                            loss_fn=criterion)
    plt.figure()
    pd.DataFrame(history).plot(figsize=(8, 5))
    plt.grid(True)
    # plt.gca().set_ylim(0, 1)
    plt.show()

if __name__ == '__main__':
    main()