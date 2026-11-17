import numpy as np, json
from lenia import *
P = dict(R=13, T=10, m=0.337, s=0.057)
def load(code):
    for x in json.load(open(ANIMALS, encoding='utf-8')):
        if x.get('code') == code and 'cells' in x: return x['params'], rle2arr(x['cells'])
if __name__ == '__main__':
    p, pat = load('[S2,PS,P4~7,H3,C0]'); print(p, pat.shape)
    N = 512; L = Lenia((N, N), **P)
    A = np.zeros((N, N)); h, w = pat.shape; A[(N-h)//2:(N-h)//2+h, (N-w)//2:(N-w)//2+w] = pat
    prev = None
    for t in range(0, 601, 100):
        bl = blobs(A, min_mass=3)
        print(t, len(bl), [(round(m), round(x), round(y)) for m, y, x in sorted(bl)])
        A = L.step(A, 100)
    np.save('fleet600.npy', A)
