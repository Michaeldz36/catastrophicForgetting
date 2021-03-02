import sys
sys.path.append("..")

from utils.utils import Student
import numpy as np


N = 1000

sgm_w0 = 1. #TODO: 0 in article..
sparsity = 1


model = Student(n_features=N, sgm_w0=sgm_w0, sparsity=sparsity)
print("our model is:", model)

if __name__ == '__main__':
    for param in model.parameters():
        np.testing.assert_array_equal(param.detach().numpy(),
                                      np.zeros((1,N)))
