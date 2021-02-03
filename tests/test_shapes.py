from utils.utils import Setup, Teacher
import numpy as np
from sklearn.model_selection import train_test_split


setup = Setup()
teacher1 = Teacher()
teacher2 = Teacher()

N1 = setup.N * 1
N2 = setup.N * 1

P1 = setup.P * 1
P2 = setup.P * 2
test_size = 0.3

X1, Y1 = teacher1.build_teacher(N1, P1, 1,1)
X2, Y2 = teacher1.build_teacher(N2, P2, 1,1)
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1,
                                                        test_size = test_size, random_state = 42)
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2,
                                                        test_size = test_size, random_state = 42)

X_train, Y_train = np.c_[X1_train.T, X2_train.T].T, np.r_[Y1_train, Y2_train]
X_test, Y_test = np.c_[X1_test.T, X2_test.T].T, np.r_[Y1_test, Y2_test]

if __name__ == '__main__':
    assert X1.shape==(P1, N1)
    assert X2.shape==(P2, N2)
    assert X1_test.shape==(int(test_size*P1),N1)
    assert X2_test.shape==(int(test_size*P2),N2)
    assert X_test.shape==(int(test_size*(P1+P2)), max(N1,N2))
    assert Y_test.shape==(int(test_size*(P1+P2)),)
    assert X_train.shape==(int((1-test_size)*(P1+P2)), max(N1, N2))
    assert Y_train.shape==(int((1-test_size)*(P1+P2)),)
