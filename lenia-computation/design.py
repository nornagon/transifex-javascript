"""Incremental design-by-search of annihilation circuits (90-degree crossings)."""
import numpy as np, json, sys
from multiprocessing import Pool
from scene import *

N = 576; ORIGIN = (60, 200)

def partner(S, target, D, i, j):
    """Position/heading of a glider that crosses `target`'s path at 90 deg after ~D cells of travel."""
    x, y, h = S.g[target]
    dx, dy = rot(h, (D+i, -D+j))
    return x+dx, y+dy, (h+90) % 360

def _try(args):
    g, target, D, i, j = args
    S = Scene(N, ORIGIN); S.g = {target: g}
    S.add('P', *partner(S, target, D, i, j))
    T = int((D+abs(i)+30)/V) + 140
    A = S.run(T)
    return (i, j), len(blobs(A, thresh=0.01, min_mass=0.2)) == 0

def find_killer(S, target, name, D, pool, I=range(8, 26), J=range(3)):
    """Add glider `name` that annihilates with pristine `target` ~D cells down its path."""
    res = dict(pool.map(_try, [(S.g[target], target, D, i, j) for j in J for i in I]))
    best = None
    for j in J:
        for i in I:
            if not res[(i, j)]: continue
            m = 0
            while res.get((i-m-1, j)) and res.get((i+m+1, j)): m += 1
            if best is None or m > best[0]: best = (m, i, j)
    print(f'{name} kills {target} at D={D}: ' + ' | '.join(''.join('.' if res[(i, j)] else 'o' for i in I) for j in J), '-> margin,i,j =', best, flush=True)
    assert best, 'no annihilating placement found'
    S.add(name, *partner(S, target, D, best[1], best[2]))
    return best

def build_xor(pool):
    S = Scene(N, ORIGIN)
    S.add('A1', 0, 0, 0)
    find_killer(S, 'A1', 'B1', 60, pool)
    find_killer(S, 'A1', 'c', 150, pool)
    find_killer(S, 'c', 'B2', 230, pool)
    find_killer(S, 'B2', 'A2', 97, pool)
    find_killer(S, 'c', 'out', 310, pool)
    return S

def truth_table(S, out='out', T=670, inputs=(('A1', 'A2'), ('B1', 'B2'))):
    rows = []
    for a in (0, 1):
        for b in (0, 1):
            pres = {n: bool(a) for n in inputs[0]}; pres.update({n: bool(b) for n in inputs[1]})
            A = S.run(T, pres)
            o = S.alive(A, out, T)
            rows.append((a, b, int(o), [(round(m, 1), round(x-S.o[0]), round(y-S.o[1])) for m, y, x in blobs(A, thresh=0.01, min_mass=0.2)]))
    return rows

if __name__ == '__main__':
    with Pool(4) as pool:
        S = build_xor(pool)
    json.dump(S.g, open('xor_circuit.json', 'w'), indent=1)
    print(S.g)
    for a, b, o, bl in truth_table(S):
        print(f'A={a} B={b} -> out={o}   remaining blobs (mass,x,y): {bl}')
