import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import Setup, Teacher, Student
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    setup = Setup()
    teacher1 = Teacher()
    teacher2 = Teacher()

    N1=setup.N * 1  # for now N=N1=N2
    N2=setup.N * 1  # TODO: make N=max(N1,N2), X=[max(X1,X2),min(X1,X2).concat(zeros)]

    P1=setup.P * 1
    P2=setup.P * 2

    sgm_w1=setup.sgm_w * 1
    sgm_w2=setup.sgm_w * 2

    X1, Y1 = teacher1.build_teacher(N1, P1, sgm_w1, setup.sgm_e)
    X2, Y2 = teacher1.build_teacher(N2, P2, sgm_w2, setup.sgm_e)
    # print(X1.shape, X2.shape)
    # print(Y1.shape, Y2.shape)
    X, Y=np.c_[X1.T,X2.T].T, np.r_[Y1,Y2]
    # print(X.shape)
    # print(Y.shape)

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, Y)

    student = Student()
    model = student.build_student(max(N1,N2), P1+P2, setup.sgm_w0)
    history = model.fit(X_train_full, y_train_full, epochs=100)

    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    # plt.gca().set_ylim(0, 1)
    plt.show()

    mse_test = model.evaluate(X_test, y_test)
