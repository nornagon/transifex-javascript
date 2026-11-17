"""Single-rail XOR from single-copy inputs: 5 Orbia, head-on collisions only (R=26).

  x  -> heading 0   on line y=0                    (input)
  y  <- heading 180 on line y=B                    (input)
  c  <- heading 180 on line y=0, trailing y by D   (constant)
  c2 -> heading 0   on line y=B, trailing x by D   (constant)
  c3    head-on along the ray on which c2 leaves after scattering off c (constant) = OUTPUT

  Head-on with lateral offset B: elastic scatter (both deflected ~37 deg).  Offset 0: annihilation.
  x and y  : x,y scatter away (two copies of x AND y); then c,c2 scatter; c3 annihilates c2'.
  only x   : x annihilates c; c2 flies straight on; c3 survives.
  only y   : y annihilates c2; c flies straight on; c3 survives.
  neither  : c,c2 scatter; c3 annihilates c2'.
  => after the c/c2 site:  c2' present iff XNOR(x,y);  c3 survives iff x XOR y.
"""
import numpy as np, json, sys
from multiprocessing import Pool
from hires import *
from scatter import fit_rays

N = 1280; ORIGIN = (400, 300)
X0, B, D, M = 110, 20, 200, 120
VG = 1.2303

def build(ray):
    S = Scene(N, ORIGIN)
    S.add('x', -X0, 0, 0); S.add('y', X0, B, 180)
    S.add('c', X0+D, 0, 180); S.add('c2', -X0-D, B, 0)
    p0 = np.array(ray['p0']); vel = np.array(ray['vel'])
    tm = (X0+D)/VG + M/VG            # c3 meets c2' M cells after the c/c2 scatter
    S.add('c3', *(p0 + vel*2*tm), ray['heading']+180)
    S.ray = (p0, vel, tm)
    return S

def case(args):
    ray, a, b, T = args
    S = build(ray)
    tr, A = S.tracks(T, present={'x': bool(a), 'y': bool(b)}, every=10)
    p0, vel, tm = S.ray
    exp = p0 + vel*(2*tm - T)            # where an undisturbed c3 would be at step T
    fin = [(x-S.o[0], y-S.o[1], m) for m, y, x in blobs(A, min_mass=20)]
    out = any(230 < m < 320 and np.hypot(x-exp[0], y-exp[1]) < 12 for x, y, m in fin)
    np.save(f'xor_final_{a}{b}.npy', A.astype(np.float32))
    return a, b, int(out), [(round(x), round(y), round(m)) for x, y, m in fin], exp.round(1).tolist(), tr

if __name__ == '__main__':
    # calibrate: outgoing ray of c2 after the c/c2 scatter (constants only, no c3)
    S = Scene(N, ORIGIN); S.add('c', X0+D, 0, 180); S.add('c2', -X0-D, B, 0)
    tr, _ = S.tracks(440, every=10)
    ray = [r for r in fit_rays(tr, 340) if r['vel'][0] > 0][0]
    print('scattered ray:', {k: np.round(v, 3).tolist() for k, v in ray.items()})
    T = 500
    with Pool(4) as p: res = p.map(case, [(ray, a, b, T) for a in (0, 1) for b in (0, 1)])
    print(build(ray).g)
    ok = True
    for a, b, o, fin, exp, tr in res:
        ok &= (o == (a ^ b))
        print(f'x={a} y={b} -> c3 alive={o} (expect {a^b});  c3 expected at {exp};  final blobs (x,y,mass): {fin}')
    print('XOR truth table', 'PASS' if ok else 'FAIL')
    json.dump(dict(gliders=build(ray).g, ray=ray, tracks={f'{a}{b}': tr for a, b, o, fin, exp, tr in res}), open('xor_hires.json', 'w'))
