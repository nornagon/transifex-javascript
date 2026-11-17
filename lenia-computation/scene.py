"""Compose Orbium circuits: gliders placed at t=0 on integer positions, axis-aligned headings.

Coordinates are (x, y), y down. Headings in degrees: 0=+x, 90=+y, 180=-x, 270=-y.
Axis-aligned gliders are exact lattice rotations of one pattern, so any two-glider reaction
found once is reproduced exactly under translation / 90-degree rotation of the pair.
"""
import numpy as np
from lenia import *
import glider as g1

V = None  # free speed (cells/step) along an axis
def _settled():
    # The freshly rotated Orbium is in a metastable gait that side-steps ~6 cells after ~450
    # steps; run it long enough to reach the final gait before cropping the canonical glider.
    n = 256; L = Lenia((n, n)); A = place(np.zeros((n, n)), g1.CANON, 128, 128)
    A = L.step(A, 1500); b = max(blobs(A))
    A = np.roll(A, (128-int(round(b[1])), 128-int(round(b[2]))), (0, 1))[108:149, 108:149]
    return A * (A > 1e-6)
P0 = _settled()        # 41x41, odd size: rot90 is exact about the centre cell

def rot(h, d):
    """Rotate integer offset d=(dx,dy) by heading h (multiple of 90), y down."""
    dx, dy = d
    return [(dx, dy), (-dy, dx), (-dx, -dy), (dy, -dx)][int(round(h / 90)) % 4]

def pat(heading):
    k = int(round(heading / 90)) % 4
    return np.rot90(P0, -k)             # y-down: rot90 with k=-1 turns +x into +y

def unit(h):
    r = np.radians(h); return np.array([np.cos(r), np.sin(r)])

class Scene:
    def __init__(self, N=512, origin=(100, 100)):
        self.N = N; self.o = np.array(origin); self.g = {}
    def add(self, name, x, y, heading):
        self.g[name] = (int(round(x)), int(round(y)), heading % 360); return self
    def field(self, present=None):
        A = np.zeros((self.N, self.N))
        for n, (x, y, h) in self.g.items():
            if present is None or present.get(n, True):
                place(A, pat(h), y + self.o[1], x + self.o[0])
        return A
    def pos(self, name, t):
        """Predicted centre of an undisturbed glider at step t (scene coords)."""
        x, y, h = self.g[name]
        return np.array([x, y]) + OFF[int(h // 90)] + t * VEL[int(h // 90)]
    def run(self, steps, present=None, snap=None):
        L = Lenia((self.N, self.N)); A = self.field(present); frames = []
        for t in range(steps):
            if snap and t % snap == 0: frames.append(A.astype(np.float32))
            A = L.step(A)
        return (A, frames) if snap else A
    def alive(self, A, name, t, tol=12.0):
        """Is there an Orbium where undisturbed glider `name` should be at step t?"""
        p = self.pos(name, t) + self.o
        for (m, y, x) in blobs(A):
            if 55 < m < 82 and np.hypot((x-p[0]+self.N/2) % self.N-self.N/2, (y-p[1]+self.N/2) % self.N-self.N/2) < tol:
                return True
        return False

def _calib():
    global V, OFF, VEL
    N = 256; L = Lenia((N, N)); OFF = []
    vs = []
    for k in range(4):
        A = place(np.zeros((N, N)), pat(90*k), 128, 128)
        A = L.step(A, 100); b0 = max(blobs(A)); tot = np.zeros(2); bp = b0
        for _ in range(20):
            A = L.step(A, 100); b1 = max(blobs(A))
            tot += [(b1[2]-bp[2]+N/2) % N-N/2, (b1[1]-bp[1]+N/2) % N-N/2]; bp = b1
        d = tot / 2000
        vs.append(d)
        OFF.append(np.array([(b0[2]-128+N/2) % N-N/2, (b0[1]-128+N/2) % N-N/2]) - 100*d)
    V = float(np.mean([np.linalg.norm(v) for v in vs]))
    VEL = vs
    return vs
_vs = _calib()

if __name__ == '__main__':
    print('V', V); print('vel', _vs); print('OFF', OFF)
