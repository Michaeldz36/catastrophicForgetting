import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import Setup, Teacher, Student, PrepareData, \
    load_data, train_valid_loop
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

setup = Setup()

### Hyperparameters
# batch_size=setup.P
lr = 1e-2
epochs1 = 500
epochs2 = 500
sgm_e = setup.sgm_e
sgm_w1 = setup.sgm_w * 1
sgm_w2 = setup.sgm_w * 2

N = 50
depth = 1 # works for small enough lr
P1 = 50
P2 = 50


def main(N=N, P1=P1, P2=P2, sgm_w1=sgm_w1, sgm_w2=sgm_w2, sgm_e=sgm_e, lr=lr, epochs1=epochs1, epochs2=epochs2, d=depth):
    teacher1 = Teacher()
    teacher2 = Teacher()

    X1, Y1 = teacher1.build_teacher(N, P1, sgm_w1, sgm_e)
    X2, Y2 = teacher2.build_teacher(N, P2, sgm_w2, sgm_e)
    # TODO: check correlation between X1, X2
    X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.33, random_state=42)
    X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=0.33, random_state=42)

    model = Student(n_features=N, sgm_e=sgm_e, depth = d)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # datasets for training the student network
    train_ds1 = PrepareData(X1_train, y=Y1_train, scale_X=True)
    train_ds2 = PrepareData(X2_train, y=Y2_train, scale_X=True)
    # datasets for validation errors
    valid_ds1 = PrepareData(X1_test, y=Y1_test, scale_X=True)
    valid_ds2 = PrepareData(X2_test, y=Y2_test, scale_X=True)
    # datasets for cross generalization error
    cross_gen_ds1 = valid_ds2
    cross_gen_ds2 = valid_ds1

    print("Lesson 1/2:")
    print('-' * 20)
    data_loaders1, data_lengths1 = load_data(train_ds=train_ds1, valid_ds=valid_ds1,
                                             generalize_ds=cross_gen_ds1, batch_size=X1_train.shape[0])

    history1 = train_valid_loop(data_loaders=data_loaders1,
                                data_lengths=data_lengths1,
                                n_epochs=epochs1,
                                optimizer=optimizer,
                                model=model,
                                criterion=criterion,
                                e_print=50
                                )
    print('Lesson 2/2:')
    print('-' * 20)
    data_loaders2, data_lengths2 = load_data(train_ds=train_ds2, valid_ds=valid_ds2,
                                             generalize_ds=cross_gen_ds2, batch_size=X2_train.shape[0])

    history2 = train_valid_loop(data_loaders=data_loaders2,
                                data_lengths=data_lengths2,
                                n_epochs=epochs2,
                                optimizer=optimizer,
                                model=model,
                                criterion=criterion,
                                e_print=50
                                )

    full_history = dict()
    for key in history1.keys():
        full_history[key] = np.array(history1[key] + history2[key])
    return full_history


def simulate(syllabus, n_runs):
    realisations = []
    for r in range(n_runs):
        print('Realisation {}/{}'.format(r, n_runs))
        history = main(*syllabus)
        realisations.append(history)
    # TODO: check (unit test), clean this clutter
    c = Counter() # sums values in lists for each computed error
    for r in realisations:
        c.update(r)
    # averaging over teacher realisations
    errors = pd.DataFrame(c) / n_runs
    return errors


def plot_history(errors, n_runs):
    errors.plot(figsize=(8, 5))
    # plt.axhline(y=sgm_e, color='r', linestyle='-')
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("Linear network with {} layers depth \n"
              "(MSE averaged over {} realisations)".format(depth, n_runs))
    textstr = '\n'.join((
        r'$N=%.2f$' % (N, ),
        r'$P_{1}=%.2f$' % (P1, ),
        r'$P_{2}=%.2f$' % (P2, ),
        r'$\sigma_{w_1}=%.2f$' % (sgm_w1,),
        r'$\sigma_{w_2}=%.2f$' % (sgm_w1,),
        r'$\sigma_{\epsilon}=%.2f$' % (sgm_e,),
        r'$\eta=%.2f$' % (lr,)
    ))
    plt.gcf().text(0.91, 0.12, textstr, fontsize=5)
    plt.show()


if __name__ == '__main__':
    syllabus = [N, P1, P2, sgm_w1, sgm_w2, sgm_e, lr, epochs1, epochs2, depth]
    n_runs = 1
    errors = simulate(syllabus, n_runs)
    plot_history(errors, n_runs)
