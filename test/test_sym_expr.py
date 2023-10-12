import sympy
import sys
sys.path.append('.')
from sympy import diff
import numpy as np
import pyzpc as zs
from pyzpc.sym import mat, sym_diff, get_sym_func_src, gen_sym_func

fsvec3_t = zs.svec_t((3, 3), zs.fl)

mat3 = mat('A', 3, 3)
# value = mat3[0, 0] ** 2 * mat3.trace()
value = mat3.trace() ** 2
value_grad = sym_diff(value, (mat3,))
gen_sym_func('value_grad', __name__, value_grad, (mat3,), elem_type=zs.fl)


@zs.kernel
def test_svec(svec0: fsvec3_t):
    idx = tid()
    grad_mat = value_grad(idx + svec0)
    print(idx, grad_mat[0, 0])


if __name__ == '__main__':
    pol = zs.Policy()
    np_arr = np.asarray([
        [1.14, 5.14, 0.],
        [1.919, 8.10, 0.], 
        [0., 0., 1.]
    ])
    svec0 = zs.svec.from_numpy(np_arr.astype(np.float32))
    zs.launch(pol, test_svec, 16, svec0)
