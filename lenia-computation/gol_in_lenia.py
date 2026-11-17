"""Conway's Life is a point in Lenia's rule space: the same update A <- clip(A + dt*G(K*A))
with R=1, a 3x3 kernel (centre weight 1/2), a rectangular growth function and dt=1."""
import numpy as np
from scipy import ndimage

K = np.array([[1, 1, 1], [1, .5, 1], [1, 1, 1.]])

def lenia_step(A):
    U = ndimage.convolve(A, K, mode='wrap')                 # potential  K * A
    G = np.where((U >= 2.5) & (U <= 3.5), 1.0, -1.0)        # growth
    return np.clip(A + 1.0 * G, 0, 1)                       # dt = 1

def life_step(A):
    n = ndimage.convolve(A, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1.]]), mode='wrap')
    return ((n == 3) | ((A == 1) & (n == 2))).astype(float)

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    A = (rng.random((128, 128)) < 0.35).astype(float); B = A.copy()
    for t in range(500):
        A, B = lenia_step(A), life_step(B)
        assert np.array_equal(A, B), t
    print('Lenia(R=1, step kernel, step growth, dt=1) == Game of Life for 500 steps on random soup: OK')


def gosper_gun(n=200):
    rle = ("24bo$22bobo$12b2o6b2o12b2o$11bo3bo4b2o12b2o$2o8bo5bo3b2o$2o8bo3bob2o4bobo$"
           "10bo5bo7bo$11bo3bo$12b2o!")
    A = np.zeros((n, n)); y = x = 0; num = ''
    for ch in rle:
        if ch.isdigit(): num += ch; continue
        k = int(num or 1); num = ''
        if ch == 'o': A[5+y, 5+x:5+x+k] = 1; x += k
        elif ch == 'b': x += k
        elif ch == '$': y += k; x = 0
    return A

if __name__ == '__main__':
    A = gosper_gun(); pops = []
    for t in range(421):
        if t % 60 == 0: pops.append(int(A.sum()))
        A = lenia_step(A)
    print('Gosper gun under the Lenia update, live cells every 60 steps:', pops, '(+10 = two new gliders per 60 steps)')
