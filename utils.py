import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler



class Setup():
    N = 500
    P = 400
    alpha = P / N

    sgm_w0 = 1.
    sgm_w = 1.5
    sgm_e = 1.2

    SNR = (sgm_w / sgm_e) ** 2
    INR = (sgm_w0 / sgm_w) ** 2


class Teacher():
    def build_teacher(self, N, P, sgm_w, sgm_e):
        np.random.seed(42) # for repruducability

        w_bar = np.random.normal(0, sgm_w, N)
        X = np.random.normal(0, np.sqrt(1 / N), [P, N])
        eps = np.random.normal(0, sgm_e, P)

        Y = X @ w_bar + eps
        # convert from float64 to float32 for various reasons (speedup, less memory usage)
        X, Y = np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
        return X, Y


class Student(nn.Module):
    def __init__(self, n_features, sgm_e=0.01, sparsity=0.):
        super(Student, self).__init__()
        self.linear = nn.Linear(in_features=n_features, out_features=1)
        nn.init.sparse_(self.linear.weight, sparsity=sparsity, std=sgm_e)

    def forward(self, x):
        return self.linear(x)


def make_ds(X,Y, scale_X=True):
    X_tensor = torch.from_numpy(X.reshape(X.shape[0], X.shape[1]))
    Y_tensor = torch.from_numpy(Y.reshape(Y.shape[0], 1))

    if scale_X:   # feature scaling
        means = X_tensor.mean(1, keepdim=True)
        deviations = X_tensor.std(1, keepdim=True)
        X_tensor -= means
        X_tensor /= deviations
    return X_tensor, Y_tensor


def training_loop(X_train,Y_train, n_epochs, optimizer, model, loss_fn):
    history={"loss":[]}
    for epoch in range(1, n_epochs + 1):
        Y_pred = model(X_train)
        loss_train = loss_fn(Y_pred, Y_train)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        history["loss"].append(loss_train.item())
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch}, Training loss {loss_train.item():.4f}")
    return history


def load_data(ds, batch_size):
    train_loader = DataLoader(ds, batch_size=batch_size,
                           sampler=SubsetRandomSampler(train))
    validation_loader = DataLoader(ds, batch_size=batch_size,
                          sampler=SubsetRandomSampler(test))
    data_loaders = {"train": train_loader, "val": validation_loader}
    return data_loaders


def train_valid_loop(data_loaders, n_epochs, optimizer, model, criterion):
    history={"E_t":[], "E_g":[]}
    for epoch in range(1, n_epochs + 1):
        print('Epoch {}/{}'.format(epoch, n_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = scheduler(optimizer, epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0

        # Iterate over data.
        for data in data_loaders[phase]:
            # get the input Xs and their corresponding Ys
            X = data['X']
            Y_true = data['Y']

            # forward pass to get outputs
            y_pred = model(X)

            # calculate the loss between predicted and target keypoints
            loss = criterion(y_pred, Y_true)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                # update the weights
                optimizer.step()

            # print loss statistics
            running_loss += loss.data[0]

        epoch_loss = running_loss / data_lengths[phase]
        print('{} Loss: {:.4f}'.format(phase, epoch_loss))
