import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class Setup():     # TODO: find good starting values
    # alpha = P / N
    N = 20
    P = 20

    # SNR = (sgm_w / sgm_e) ** 2
    # INR = (sgm_w0 / sgm_w) ** 2
    sgm_w = 1.
    sgm_w0 = 0.
    sgm_e = 0.2




class Teacher(): #TODO: check if redundant
    def build_teacher(self, N, P, sgm_w, sgm_e):
        # np.random.seed(42) # for repruducability
        w_bar = np.random.normal(0, sgm_w, N)
        X = np.random.normal(0, np.sqrt(1 / N), [P, N])
        eps = np.random.normal(0, sgm_e, P)

        Y = X @ w_bar + eps
        # convert from float64 to float32 for various reasons (speedup, less memory usage)
        X, Y = np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
        return X, Y, w_bar


class Student(nn.Module):
    def __init__(self, n_features, sgm_w0=0.01, sparsity=0., depth = 1):
        super(Student, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(in_features=n_features, out_features=n_features, bias=False) for _ in range(depth-1)])
        self.fc = nn.Linear(in_features=n_features, out_features=1, bias=False)
        for i in range(depth-1):
            nn.init.sparse_(self.linears[i].weight, sparsity=sparsity, std=sgm_w0)
        nn.init.sparse_(self.fc.weight, sparsity=sparsity, std=sgm_w0)

    def forward(self, x):
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        output = self.fc(x)
        return output


def make_ds(X,Y, scale_X=True):
    X_tensor = torch.from_numpy(X.reshape(X.shape[0], X.shape[1]))
    Y_tensor = torch.from_numpy(Y.reshape(Y.shape[0], 1))

    if scale_X:   # feature scaling
        means = X_tensor.mean(1, keepdim=True)
        deviations = X_tensor.std(1, keepdim=True)
        X_tensor -= means
        X_tensor /= deviations
    return (X_tensor, Y_tensor)


class PrepareData(Dataset):
    def __init__(self, X, y, scale_X=True):
        if not torch.is_tensor(X):
            if scale_X:
                X = StandardScaler().fit_transform(X)
            self.X = torch.from_numpy(X)
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(train_ds, valid_ds=None, generalize_ds=None, batch_size=1):
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                           sampler=None)
    validation_loader = DataLoader(valid_ds, batch_size=batch_size,
                          sampler=None)
    gen_loader = DataLoader(generalize_ds, batch_size=batch_size,
                          sampler=None)

    data_loaders = {"train": train_loader, "val": validation_loader, "cross_gen": gen_loader}
    data_lengths = {"train": len(train_ds), "val": len(valid_ds), "cross_gen": len(generalize_ds)}
    return data_loaders, data_lengths


def train_valid_loop(data_loaders, data_lengths, n_epochs,
                     optimizer, model, criterion, e_print=1):
    history={"E_train":[], "E_valid":[], "E_cross_g":[]}
    if n_epochs == 0:
        return history
    for epoch in range(1, n_epochs + 1):
        if epoch % e_print == 0:
            print('Epoch {}/{}'.format(epoch, n_epochs))
            print('-' * 10)
        ### Each epoch has a training and validation phase
        phases = ['train', 'val', 'cross_gen']
        for phase in phases:
            if phase == 'train':
                model.train(True)  ### Set model to training mode
            else:
                model.train(False)  ### Set model to evaluate mode

            running_loss = 0.0

            ### Iterate over data.
            for (Xb, yb) in data_loaders[phase]:
                ### get the input Xs and their corresponding Ys
                X = Xb
                Y_true = torch.reshape(yb, (yb.shape[0], 1))

                ### forward pass to get outputs
                y_pred = model(X)


                ### calculate the loss between predicted and target
                loss = criterion(y_pred, Y_true)
                if loss.item()==float("inf"):
                    print("Possibly too large lr!")
                    raise ValueError
                ### zero the parameter (weight) gradients
                optimizer.zero_grad()

                ### backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # update the weights
                    optimizer.step()

                ### update running loss
                running_loss += loss.item()
            # TODO!!!! proper normalisation for batch learning!!!!!
            epoch_loss = running_loss #/ data_lengths[phase]
            if phase == 'train':
                history["E_train"].append(epoch_loss)
            elif phase == 'val':
                history["E_valid"].append(epoch_loss)
            elif phase == 'cross_gen':
                history["E_cross_g"].append(epoch_loss)
            if epoch % e_print == 0:
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
        if epoch % e_print == 0:
            print('\n')
    return history

def training_loop(X_train, Y_train, n_epochs, optimizer, model, loss_fn):
    #todo: legacy function, not used in the main simulation, generalize train_valid_loop
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


