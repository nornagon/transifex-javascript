"""R=26 Lenia (Orbium at 2x spatial resolution): scenes with arbitrary headings.
Coordinates (x, y) in cells, y down; headings in degrees (0=+x, 90=+y)."""
import numpy as np, os
from scipy import ndimage
from lenia import *
import glider as g1

R = 26
HERE = os.path.dirname(os.path.abspath(__file__))

def _canon():
    f = os.path.join(HERE, 'orbium_R26.npy')
    if os.path.exists(f): return np.load(f)
    p = np.clip(ndimage.zoom(g1.CANON, 2, order=3), 0, 1)
    n = 512; L = Lenia((n, n), R=R)
    A = L.step(place(np.zeros((n, n)), p, n//2, n//2), 1500)
    b = max(blobs(A, min_mass=20))
    A = np.roll(A, (n//2-int(round(b[1])), n//2-int(round(b[2]))), (0, 1))[n//2-40:n//2+41, n//2-40:n//2+41]
    A = A*(A > 1e-6); np.save(f, A); return A
P0 = _canon()                       # 81x81, heading ~ +x

def pat(h):
    if h % 360 == 0: return P0
    if h % 360 == 180: return P0[::-1, ::-1]
    return np.clip(ndimage.rotate(P0, -h, reshape=False, order=3), 0, 1)

def unit(h):
    r = np.radians(h); return np.array([np.cos(r), np.sin(r)])

def gliders(A):
    """Orbium-like blobs -> list of (x, y, mass)."""
    return [(x, y, m) for m, y, x in blobs(A, min_mass=20) if 230 < m < 320]

class Scene:
    def __init__(self, N, origin):
        self.N = N; self.o = np.array(origin, float); self.g = {}
    def add(self, name, x, y, h):
        self.g[name] = (int(round(x)), int(round(y)), h % 360); return self
    def field(self, present=None):
        A = np.zeros((self.N, self.N))
        for n, (x, y, h) in self.g.items():
            if present is None or present.get(n, True):
                place(A, pat(h), y+self.o[1], x+self.o[0])
        return A
    def run(self, steps, present=None, every=None, cb=None):
        L = Lenia((self.N, self.N), R=R); A = self.field(present)
        for t in range(steps):
            if every and t % every == 0: cb(t, A)
            A = L.step(A)
        return A
    def tracks(self, steps, present=None, every=10):
        """-> {t: [(x,y,mass)...]} in scene coords (all blobs)."""
        out = {}
        def cb(t, A): out[t] = [(x-self.o[0], y-self.o[1], m) for m, y, x in blobs(A, min_mass=20)]
        A = self.run(steps, present, every, cb)
        return out, A

if __name__ == '__main__':
    N = 1024; S = Scene(N, (100, 512)); S.add('g', 0, 0, 0)
    tr, _ = S.tracks(1400, every=100)
    for t, b in tr.items(): print(t, [(round(x, 1), round(y, 1), round(m, 1)) for x, y, m in b])
