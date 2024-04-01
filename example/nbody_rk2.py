import pyzpc as zs
from pyzpc import svec_t, svec

# container types
tv_t = zs.tv_t(zs.fl)    # zs.dbl
vec_t = zs.v_t(zs.int)
# view types
tvv_t = tv_t.view_t(named=False)
tvnv_t = tv_t.view_t(named=True)
vv_t = vec_t.view_t()
# small tensors
svec3 = zs.svec_t((3,), zs.fl)
svec2 = zs.svec_t((2,), zs.fl)
smat2 = zs.svec_t((2, 2,), zs.fl)

@zs.kernel
def init_tv(tv: tvnv_t):
    idx = tid()
    tv['m', 0, idx] = 1
    tv[(3, ), 'pos', idx] = svec3(idx, 0, idx % 10)
    tv[(3, ), 'vel', idx] = svec3(0, 0, 0)

@zs.kernel
def apply_force(n: zs.int, tv: tvnv_t, dt: zs.fl):
    G = 6.67430e-11

    i = tid()
    mi = tv['m', 0, i]
    pi = tv[(3, ), 'pos', i]
    f = svec3(0, 0, 0)
    for j in range(n):
        if i != j:
            mj = tv['m', 0, j]
            pj = tv[(3, ), 'pos', j]
            d = (pj - pi)
            dist = d.length() + 1e-7
            f += G * mi * mj / dist / dist * (d / dist)
    dv = f / mi * dt
    tv[(3,), 'vel', i] += dv

@zs.kernel
def update_states(tv: tvnv_t, dt: zs.fl):
    idx = tid()
    tv[(3, ), 'pos', idx] += tv[(3, ), 'vel', idx] * dt

if __name__ == '__main__':
    N = 4096
    dt = 1e-4
    tv = tv_t('device', {'mass': 1, 'pos': 3, 'vel': 3}, N)
    # vec = vec_t('device', N)
    pol = zs.Policy()
    zs.launch(pol, init_tv, N, tv)
    for stepi in range(100):
        zs.launch(pol, apply_force, N, N, tv, dt * 0.5)
        zs.launch(pol, update_states, N, tv, dt)
        zs.launch(pol, apply_force, N, N, tv, dt * 0.5)
    print("done compute n-body using RK2")
