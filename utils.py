import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn
import torch.optim as optim

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

# class Student():
    # def build_student(self, N, P, sgm_w0):
    #     initializer = keras.initializers.RandomNormal(mean=0., stddev=sgm_w0)
    #     model = keras.models.Sequential([
    #         keras.layers.Flatten(input_shape=[N, ]),
    #         keras.layers.Dense(P, activation=None, name="Final", kernel_initializer=initializer),
    #     ])
    #     model.compile(loss = "mean_squared_error", optimizer ='sgd')       # model.summary()
    #     return model


class Student(nn.Module):
    def __init__(self, n_features, sgm_e=0.01, sparsity=0.):
        super(Student, self).__init__()
        self.linear = nn.Linear(in_features=n_features, out_features=1)
        nn.init.sparse_(self.linear.weight, sparsity=sparsity, std=sgm_e)

    def forward(self, x):
        return self.linear(x)
