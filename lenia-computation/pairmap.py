"""Outcome map for a 90-degree crossing: A at (0,0) heading 0, B at (D+i, -D+j) heading 90."""
import numpy as np, sys, json
from multiprocessing import Pool
from scene import *

def outcome(args):
    D, i, j, hB = args
    N = 256; S = Scene(N, origin=(60, 128))
    S.add('A', 0, 0, 0)
    if hB == 90: S.add('B', D+i, -D+j, 90)
    else: S.add('B', D+i, D+j, 270)
    T = int((D+abs(i)+40)/V) + 150
    A = S.run(T)
    bl = blobs(A, thresh=0.02, min_mass=0.5)
    a = S.alive(A, 'A', T); b = S.alive(A, 'B', T)
    if not bl: code = '.'
    elif a and b and len(bl) == 2: code = '='        # passed through untouched
    else:
        n = sum(1 for m, _, _ in bl if 55 < m < 82)
        code = str(n) if n == len(bl) else 'x'
    return (D, i, j, hB), code

if __name__ == '__main__':
    D = int(sys.argv[1]); hB = int(sys.argv[2])
    I = range(-36, 37); J = range(3)
    with Pool(4) as p: res = dict(p.map(outcome, [(D, i, j, hB) for j in J for i in I]))
    print('i from', I[0], 'to', I[-1], " ('.'=annihilation, '='=no interaction, n=n gliders, x=other)")
    for j in J: print(f'D={D} hB={hB} j={j}: ' + ''.join(res[(D, i, j, hB)] for i in I))
