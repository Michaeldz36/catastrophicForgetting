import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from utils.utils import Student, train_valid_loop, PrepareData, load_data

N=3
sgm_w0=1
sparsity=1
d=1

lr=1e-2

model = Student(n_features=N, sgm_w0=sgm_w0, sparsity=sparsity, depth=d)
optimizer = optim.SGD(model.parameters(), lr=lr)
criterion = nn.MSELoss(reduction='mean')

X_train=np.array([[1,1,1],[2,2,2],[3,3,3]])
Y_train=np.array([2,4,6])

X_valid=np.array([[2,2,2],[3,3,3],[4,4,4]])
Y_valid=np.array([4,6,8])

print("X_train.shape", X_train.shape)
print("X_test.shape", X_valid.shape)
print("Y_train.shape", Y_train.shape)
print("Y_test.shape", Y_valid.shape)

train_ds = PrepareData(X_train, y=Y_train, scale_X=False)
valid_ds = PrepareData(X_valid, y=Y_valid, scale_X=False)

data_loaders, data_lengths = load_data(train_ds=train_ds, valid_ds=valid_ds,
                                        batch_size=X_train.shape[0])

for phase in ['train','valid']:
    print("\nPHASE IS: {}".format(phase))
    for (Xb, yb) in data_loaders[phase]:
        print("X in batch is \n{}".format(Xb))
        print("Y in batch is \n{}".format(yb))

if __name__ == '__main__':
    h=train_valid_loop(data_loaders=data_loaders,
                     data_lengths=data_lengths,
                     n_epochs=2,
                     optimizer=optimizer,
                     model=model,
                     criterion=criterion,
                     e_print=1,
                     phases=['train', 'valid',]
                     )
    print(h)
