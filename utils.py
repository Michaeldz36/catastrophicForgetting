import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np

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
        w_bar = np.random.normal(0, sgm_w, N)
        X = np.random.normal(0, np.sqrt(1 / N), [N, P])
        eps = np.random.normal(0, sgm_e, P)

        Y = w_bar @ X + eps
        return X.T, Y

class Student():
    def build_student(self, N, P, sgm_w0):
        initializer = keras.initializers.RandomNormal(mean=0., stddev=sgm_w0)
        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=[N, ]),
            keras.layers.Dense(P, activation=None, name="Final", kernel_initializer=initializer),
        ])
        # hidden1 = model.layers[1]
        # weights, biases = hidden1.get_weights()
        model.compile(loss = "mean_squared_error", optimizer = keras.optimizers.SGD(lr=1e-2))       # model.summary()
        return model

