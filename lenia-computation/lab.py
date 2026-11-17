"""Scale-aware collision lab. Lengths are given in 'u' = cells at R=13; scale s -> R=13*s."""
import numpy as np, sys, json
from multiprocessing import Pool
from scipy import ndimage
from lenia import *
import glider as g1

_cache = {}
def canon(s):
    if s not in _cache:
        if s == 1: _cache[s] = g1.CANON
        else:
            p = np.clip(ndimage.zoom(g1.CANON, s, order=3), 0, 1)
            # settle on the fine grid
            n = 96*s; L = Lenia((n, n), R=13*s)
            A = L.step(place(np.zeros((n, n)), p, n//2, n//4), 60)
            b = max(blobs(A, min_mass=5*s*s))
            h = 20*s
            A = np.roll(A, (n//2-int(round(b[1])), n//2-int(round(b[2]))), (0, 1))[n//2-h:n//2+h, n//2-h:n//2+h]
            _cache[s] = A*(A > 1e-4)
    return _cache[s]

def glider(angle, s=1):
    p = canon(s)
    if angle % 360 == 0: return p
    return np.clip(ndimage.rotate(p, -angle, reshape=False, order=3), 0, 1)

def track(L, A, s, n=30):
    N = A.shape[0]
    b0 = blobs(A, min_mass=5*s*s); A2 = L.step(A, n); b1 = blobs(A2, min_mass=5*s*s)
    out = []
    for (m, y, x) in b1:
        if not b0: break
        d = [np.hypot((y-y0+N/2) % N-N/2, (x-x0+N/2) % N-N/2) for (_, y0, x0) in b0]
        j = int(np.argmin(d)); m0, y0, x0 = b0[j]
        dy = (y-y0+N/2) % N-N/2; dx = (x-x0+N/2) % N-N/2
        out.append(dict(mass=round(m/s/s, 1), y=round(y/s, 1), x=round(x/s, 1),
                        heading=round(float(np.degrees(np.arctan2(dy, dx))) % 360, 1),
                        speed=round(float(np.hypot(dy, dx)/n/s), 3), dm=round((m-m0)/s/s, 1)))
    return out

def is_glider(o):
    return 60 < o['mass'] < 78 and 0.5 < o['speed'] < 0.7 and abs(o['dm']) < 4

def summarize(out, phi=0):
    g = [o for o in out if is_glider(o)]
    other = [o for o in out if not is_glider(o)]
    s = f"{len(g)}[" + ",".join(f"{(o['heading']-phi)%360:.0f}" for o in sorted(g, key=lambda o: (o['heading']-phi) % 360)) + "]"
    if other: s += "+X"
    return s

def collide(theta, bs, phi=0, lead=0.0, s=1, T_after=170, sep=48, Nu=256):
    """A heads phi through centre, B heads phi+theta, displaced perpendicular by b.
    Both start far enough back that their initial centre distance is >= sep. `lead` shifts
    both back further (changes sub-cell phase at impact)."""
    N = Nu*s; C = N/2
    L = Lenia((N, N), R=13*s)
    A = np.zeros((len(bs), N, N))
    th = np.radians(theta+phi); ph = np.radians(phi)
    uA = np.array([np.cos(ph), np.sin(ph)]); uB = np.array([np.cos(th), np.sin(th)]); nB = np.array([-np.sin(th), np.cos(th)])
    Dmax = 0
    for i, b in enumerate(bs):
        D = 30.0
        while np.linalg.norm((-D*uA) - (-D*uB + b*nB)) < sep: D += 1
        D += lead; Dmax = max(Dmax, D)
        pa = -D*uA; pb = -D*uB + b*nB
        place(A[i], glider(phi, s), C+pa[1]*s, C+pa[0]*s)
        place(A[i], glider(theta+phi, s), C+pb[1]*s, C+pb[0]*s)
    A = L.step(A, int(Dmax/g1.SPEED) + T_after)
    return [track(L, A[i], s) for i in range(len(bs))]

def job(a):
    return a, collide(*a[:4], **a[4])

if __name__ == '__main__':
    s = int(sys.argv[1]); thetas = [int(x) for x in sys.argv[2].split(',')]
    bs = np.arange(-30, 31, 2.0)
    jobs = [(t, bs, phi, lead, dict(s=s)) for t in thetas for phi in (0, 20) for lead in (0, 3.3, 6.7)]
    res = []
    with Pool(4) as p:
        for a, rs in p.imap(job, jobs):
            print(f"th={a[0]} phi={a[2]} lead={a[3]}: " + ' '.join(f"{b:+.0f}:{summarize(r, a[2])}" for b, r in zip(bs, rs)), flush=True)
            res.append(dict(theta=a[0], phi=a[2], lead=a[3], bs=list(bs), out=rs))
    json.dump(res, open(f'lab_s{s}.json', 'w'))
