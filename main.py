import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import Setup, Teacher, Student
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    setup = Setup()
    teacher = Teacher()
    student = Student()

    # X, Y = teacher.build_teacher(setup.N, setup.P, setup.sgm_w, setup.sgm_e)
    # testing if it works with random data
    X, Y = np.random.normal(0, np.sqrt(1 / setup.N), [setup.P, setup.N]), np.random.normal(0, np.sqrt(1 / setup.P), [setup.P])
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, Y)
    model = student.build_student(setup.N, setup.P, setup.sgm_w0)
    history = model.fit(X_train_full, y_train_full, epochs=50, batch_size=setup.P)

    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    # plt.gca().set_ylim(0, 1)
    plt.show()

    mse_test = model.evaluate(X_test, y_test)
