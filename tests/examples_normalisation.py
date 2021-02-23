import numpy as np
from utils.utils import Teacher

N=1000
P=10000
sgm_w=1
sgm_e=0.1

teacher = Teacher()
X, _, _ = teacher.build_teacher(N, P, sgm_w, sgm_e)
L2_norms = np.linalg.norm(X, axis=1)
average = np.mean(L2_norms)
if __name__ == '__main__':
    np.testing.assert_almost_equal(average, 1., 3)
    print("Data properly normalised")