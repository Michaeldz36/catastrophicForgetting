import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import Setup, Teacher, Student, training_loop

### Hyperparameters
learning_rate = 1e-2
epochs = 500
scaling=True

if __name__ == '__main__':
    setup = Setup()
    teacher1 = Teacher()
    teacher2 = Teacher()

    N1 = setup.N * 1  # for now N=N1=N2
    N2 = setup.N * 1  # TODO: make N=max(N1,N2), X=[max(X1,X2),min(X1,X2).concat(zeros)]

    P1 = setup.P * 1
    P2 = setup.P * 2

    sgm_w1 = setup.sgm_w * 1
    sgm_w2 = setup.sgm_w * 2

    X1, Y1 = teacher1.build_teacher(N1, P1, sgm_w1, setup.sgm_e)
    X2, Y2 = teacher1.build_teacher(N2, P2, sgm_w2, setup.sgm_e)

    X, Y = np.c_[X1.T, X2.T].T, np.r_[Y1, Y2]

    X_tensor = torch.from_numpy(X.reshape(P1+P2, setup.N))
    Y_tensor = torch.from_numpy(Y.reshape(P1+P2, 1))

    N_FEATURES = setup.N
    model = Student(n_features=N_FEATURES, sgm_e=setup.sgm_e)

    # feature scaling
    means = X_tensor.mean(1, keepdim=True)
    deviations = X_tensor.std(1, keepdim=True)
    X_scaled = (X_tensor - means) / deviations
    if scaling:
        X_tensor=X_scaled

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    history = training_loop(X_train=X_tensor, Y_train=Y_tensor, n_epochs=epochs,
                            optimizer=optimizer,
                            model=model,
                            loss_fn=criterion)

    pd.DataFrame(history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.show()