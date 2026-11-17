"""Fleet world: fire a Discutium glider (heading -y) at a stationary target, vs lateral offset b."""
import numpy as np, sys, json
from multiprocessing import Pool
from world import *

def settled(name, steps=400):
    N = 256; L = sim(N); A = L.step(place(np.zeros((N, N)), SP[name], 128, 128), steps)
    m, y, x = max(blobs(A, min_mass=3))
    A = np.roll(A, (128-int(round(y)), 128-int(round(x))), (0, 1))[88:169, 88:169]
    return A*(A > 1e-6)

_c = {}
def S(name):
    if name not in _c: _c[name] = settled(name)
    return _c[name]

def shot(args):
    target, bs, D, T = args
    N = 384; L = sim(N); A = np.zeros((len(bs), N, N)); C = N//2
    for k, b in enumerate(bs):
        if target != 'none': place(A[k], S(target), C, C)
        place(A[k], S('m260'), C+D, C+b)
    snaps = {}
    for t in range(T+1):
        if t in (T-100, T): snaps[t] = [[(round(m), round(x-C, 1), round(y-C, 1)) for m, y, x in blobs(A[k], min_mass=3)] for k in range(len(bs))]
        A = L.step(A)
    return [(int(b), snaps[T-100][k], snaps[T][k]) for k, b in enumerate(bs)]

if __name__ == '__main__':
    target = sys.argv[1]; bs = list(range(-50, 51, 2))
    chunks = [bs[i::4] for i in range(4)]
    with Pool(4) as p: res = sum(p.map(shot, [(target, c, 90, 620) for c in chunks]), [])
    res.sort()
    json.dump(res, open(f'wsurvey_{target}.json', 'w'))
    for b, s0, s1 in res: print(b, s0, '=>', s1)
