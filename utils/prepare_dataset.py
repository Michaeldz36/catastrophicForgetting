import numpy as np
from utils.utils import Setup, Teacher
from sklearn.model_selection import train_test_split



setup = Setup()
teacher1 = Teacher()
teacher2 = Teacher()

# TODO: glue it with make_ds from utils
def make_ds_WIP():
    N = setup.N

    P1 = setup.P * 1
    P2 = setup.P * 2

    sgm_w1 = setup.sgm_w * 1
    sgm_w2 = setup.sgm_w * 2

    X1, Y1 = teacher1.build_teacher(N, P1, sgm_w1, setup.sgm_e)
    X2, Y2 = teacher1.build_teacher(N, P2, sgm_w2, setup.sgm_e)
    X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size = 0.33, random_state = 42)
    X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size = 0.33, random_state = 42)

    X_train, Y_train = np.c_[X1_train.T, X2_train.T].T, np.r_[Y1_train, Y2_train]
    X_test, Y_test = np.c_[X1_test.T, X2_test.T].T, np.r_[Y1_test, Y2_test]
    return X_train, Y_train, X_test, Y_test