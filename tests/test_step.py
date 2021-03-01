import torch
import torch.optim as optim
import torch.nn as nn
from utils.utils import Student, train_valid_loop, PrepareData, load_data

N=3
sgm_w0=1e-10
sparsity=1
d=1

lr=1e-3

model = Student(n_features=N, sgm_w0=sgm_w0, sparsity=sparsity, depth=d)
optimizer = optim.SGD(model.parameters(), lr=lr)
criterion = nn.MSELoss()

X_train=torch.tensor([[1,0,0],[0,0,0],[0,0,0]])
Y_train=None
X_valid=None
Y_valid=None

train_ds = PrepareData(X_train, y=Y_train, scale_X=False)
# datasets for validation errors
valid_ds = PrepareData(X_valid, y=Y_valid, scale_X=False)
# datasets for cross generalization error
cross_gen_ds = valid_ds

if __name__ == '__main__':
    h=train_valid_loop(data_loaders=data,
                                data_lengths=3,
                                n_epochs=1,
                                optimizer=optimizer,
                                model=model,
                                criterion=criterion,
                                e_print=1)
    print(h)