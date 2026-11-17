"""What else can two colliding Orbia make (R=13)? Scan 90-degree crossings and log every
outcome that is not just 0/1/2 clean gliders."""
import numpy as np, sys, json
from multiprocessing import Pool
from scene import *

N = 384
def run(args):
    D, i, j, hB = args
    S = Scene(N, (100, 192)); S.add('A', 0, 0, 0)
    if hB == 90: S.add('B', D+i, -D+j, 90)
    elif hB == 270: S.add('B', D+i, D+j, 270)
    else: S.add('B', 2*D+j, i, 180)
    T = int((D+40)/V) + 260
    A = S.run(T)
    bl = sorted(round(m, 1) for m, _, _ in blobs(A, thresh=0.02, min_mass=0.5))
    return args, bl, round(float(A.sum()), 1)

if __name__ == '__main__':
    jobs = [(60, i, j, h) for h in (90, 180) for j in range(3) for i in range(-30, 31)]
    with Pool(4) as p: res = p.map(run, jobs)
    json.dump(res, open('products.json', 'w'))
    from collections import Counter
    c = Counter()
    for a, bl, tot in res:
        ng = sum(1 for m in bl if 60 < m < 78); other = [m for m in bl if not 60 < m < 78]
        c[(ng, bool(other))] += 1
        if other or ng > 2: print(a, 'gliders', ng, 'other', other, 'total', tot)
    print(c)
