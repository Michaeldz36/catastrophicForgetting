import pandas as pd
import matplotlib.pyplot as plt
from utils import Setup, Teacher, Student
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim



def training_loop( X_train,Y_train, n_epochs, optimizer, model, loss_fn):
    history=[]
    for epoch in range(1, n_epochs + 1):
        Y_pred = model(X_train)
        loss_train = loss_fn(Y_pred, Y_train)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        history.append(loss_train.item())
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch}, Training loss {loss_train.item():.4f}")
    return history

if __name__ == '__main__':
    setup = Setup()
    teacher = Teacher()

    X, Y = teacher.build_teacher(setup.N, setup.P, setup.sgm_w, setup.sgm_e)
    # for pytorch, shape (m,1) i.e. (100x1) matrix, is crucial.
    # A vector with shape (100,) will mess up calculations.
    X_tensor = torch.from_numpy(X.reshape(setup.P, setup.N))
    Y_tensor = torch.from_numpy(Y.reshape(setup.P, 1))

    N_FEATURES = setup.N
    model = Student(n_features=N_FEATURES)
    y_predicted = model(X_tensor)

    # feature scaling
    means = X_tensor.mean(1, keepdim=True)
    deviations = X_tensor.std(1, keepdim=True)
    X_scaled = (X_tensor - means) / deviations
    X_tensor=X_scaled

    ### Hyperparameters
    learning_rate = 1e-3
    epochs = 1000
    costs = []
    mean = []
    std = []

    criterion = nn.MSELoss()
    # loss = criterion(y_tensor, predictions)
    optimizer = optim.SGD(model.parameters(), lr = learning_rate)

    history = training_loop(X_train=X_tensor, Y_train=Y_tensor, n_epochs=epochs,
                            optimizer=optimizer,
                            model=model,
                            loss_fn=criterion)

    pd.DataFrame(history).plot(figsize=(8, 5))
    plt.grid(True)
    # plt.gca().set_ylim(0, 1)
    plt.show()