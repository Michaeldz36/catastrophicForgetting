import torch
import torch.optim as optim
import torch.nn as nn
from utils.utils import Student, train_valid_loop
#TODO: WIP!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

N=3
sgm_w0=1e-10
sparsity=1
d=1

lr=1e-3

model = Student(n_features=N, sgm_w0=sgm_w0, sparsity=sparsity, depth=d)
optimizer = optim.SGD(model.parameters(), lr=lr)
criterion = nn.MSELoss()

data=torch.tensor([[1,2,3],[2,4,6],[3,6,9]])

if __name__ == '__main__':
    h=train_valid_loop(data_loaders=data,
                                data_lengths=3,
                                n_epochs=1,
                                optimizer=optimizer,
                                model=model,
                                criterion=criterion,
                                e_print=1)
    print(h)