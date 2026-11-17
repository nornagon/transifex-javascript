"""The 'fleet' Lenia world (R=13, T=10, m=0.337, s=0.057): gliders and stationary creatures coexist."""
import numpy as np, os
from lenia import *
P = dict(R=13, T=10, m=0.337, s=0.057)
HERE = os.path.dirname(os.path.abspath(__file__))

def _extract():
    f = os.path.join(HERE, 'fleet_species.npz')
    if os.path.exists(f): return dict(np.load(f))
    A = np.load(os.path.join(HERE, 'fleet600.npy')); N = A.shape[0]
    L = Lenia((N, N), **P); A = L.step(A, 400)
    out = {}
    for m, y, x in blobs(A, min_mass=3):
        r = 30
        ys = (np.arange(-r, r+1)+int(round(y))) % N; xs = (np.arange(-r, r+1)+int(round(x))) % N
        yy, xx = np.mgrid[-r:r+1, -r:r+1]
        lab, n = ndimage.label(A[np.ix_(ys, xs)] > 0.01)
        c = A[np.ix_(ys, xs)] * (lab == lab[r, r] if lab[r, r] else 1)
        out[f'm{int(round(m))}'] = c
    np.savez(f, **out); return out
from scipy import ndimage
SP = _extract()

def sim(N=384): return Lenia((N, N), **P)

if __name__ == '__main__':
    for k, v in SP.items(): print(k, v.shape, round(v.sum(), 1))
    # isolate each and check survival / velocity
    N = 256; L = sim(N)
    for k, v in SP.items():
        A = place(np.zeros((N, N)), v, 128, 128); A = L.step(A, 200); b0 = blobs(A, min_mass=3); A = L.step(A, 200); b1 = blobs(A, min_mass=3)
        print(k, [(round(m), round(x, 1), round(y, 1)) for m, y, x in b0], '->', [(round(m), round(x, 1), round(y, 1)) for m, y, x in b1])
