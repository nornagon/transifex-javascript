"""Scan (m, s) around Catenopteryx increscens: does the growing chain ever shed pieces?"""
import numpy as np, json, sys
from multiprocessing import Pool
from lenia import *
a = json.load(open(ANIMALS, encoding='utf-8'))
PAT = rle2arr([x for x in a if x.get('code') == 'PN+i' and 'cells' in x][0]['cells'])

def run(ms):
    m, s = ms; N = 640; L = Lenia((N, N), R=13, T=10, m=m, s=s)
    A = np.zeros((N, N)); h, w = PAT.shape; A[(N-h)//2:(N-h)//2+h, (N-w)//2:(N-w)//2+w] = PAT
    hist = []
    for t in range(0, 1601, 400):
        bl = sorted((round(mm) for mm, _, _ in blobs(A, min_mass=20)), reverse=True)
        hist.append((t, len(bl), round(float(A.sum())), bl[:6]))
        if A.sum() > 30000 or A.sum() < 1: break
        A = L.step(A, 400)
    return (m, s), hist

if __name__ == '__main__':
    ms = [(round(m, 4), round(s, 4)) for m in np.arange(0.26, 0.3401, 0.01) for s in np.arange(0.036, 0.0501, 0.001)]
    with Pool(4) as p: res = p.map(run, ms)
    json.dump(res, open('chainscan.json', 'w'))
    def code(h):
        t, n, tot, bl = h[-1]
        if tot < 1: return '.'
        if tot > 30000: return '#'
        if n == 1: return 'G' if h[-1][2] > 1.5*h[0][2] else '1'
        return str(min(n, 9))
    print("legend: .=dies  #=explodes  1=single stable  G=single growing  n=number of separate blobs at end")
    ss = sorted({s for (m, s), _ in res})
    print('       s=' + ' '.join(f'{s*1000:.0f}' for s in ss))
    for m in sorted({m for (m, s), _ in res}):
        print(f'm={m:.2f}  ' + '  '.join(code(h) for (mm, s), h in res if mm == m))
