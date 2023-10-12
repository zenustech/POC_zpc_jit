import numpy as np
from pyzpc import svec_t, svec
import pyzpc as zs
import sys
sys.path.append('.')


tv_t = zs.tv_t(zs.dbl)
tvv_t = tv_t.view_t(named=False)
tvnv_t = tv_t.view_t(named=True)
vec_t = zs.v_t(zs.int)
vv_t = vec_t.view_t()
fsvec2_t = svec_t((2, 2), zs.fl)


@zs.func
def test_sqr(x):
    x = x * x


@zs.func
def test_func(a, b):
    a = b * 2
    for i in range(6):
        b += a * i
    return b


@zs.llvm_kernel
def init_tv(tv: tvnv_t, vec: vv_t):
    idx = tid
    vec[idx] = idx + idx % 2
    tv['z', 2, idx] = idx
    if idx % 256 == 0:
        print('[init vec]', idx, vec[idx])
        print(idx, tv['z', 2, idx])


@zs.llvm_kernel
def test_tv(tv: tvv_t, vec: vv_t):
    idx = tid
    if idx % 256 == 0:
        print('before_test_func', idx, tv['z', 2, idx])
        var = test_func(tv['z', 2, idx], 1)
        print(idx, tv['z', 2, idx], var)
        test_sqr(vec[idx])
        print('[show vec]', idx, vec[idx])


if __name__ == '__main__':
    tv = tv_t('host', {'a': 1, 'z': 3}, 1024)
    vec = vec_t('host', 1024)
    pol = zs.Policy()
    zs.launch_llvm(pol, init_tv, 1024, tv, vec)
    zs.launch_llvm(pol, test_tv, 1024, tv, vec)
    size_one_vec = vec_t('host', 1)
    size_one_vec.set_val(114514)
    print(size_one_vec.get_val())
    vec.resize(2048)
    tv.resize(2048)
    zs.launch_llvm(pol, init_tv, 2048, tv, vec)
    zs.launch_llvm(pol, test_tv, 2048, tv, vec)

    np_arr = np.asarray([
        [1.14, 5.14],
        [1.919, 8.10]
    ])
    svec0 = svec.from_numpy(np_arr.astype(np.float32))
    test_np_arr = svec0.to_numpy()
    print(test_np_arr)
