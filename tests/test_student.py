from utils.utils import Student
import torch.nn as nn
import torch.optim as optim

### Hyperparameters
N = 3
P = 4

sgm_w0 = 1. #TODO: 0 in article..

lr=1e-3

model = Student(n_features=N, sgm_w0=sgm_w0, depth = 1)
print("our model is:", model)
optimizer = optim.SGD(model.parameters(), lr=lr)
criterion = nn.MSELoss()

if __name__ == '__main__':
