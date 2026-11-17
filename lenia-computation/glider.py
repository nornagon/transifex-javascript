"""Canonical Orbium heading +x, and helpers to place gliders with a given heading."""
import numpy as np
from scipy import ndimage
from lenia import *

SPEED = None

def velocity(L, A, n=20):
    b0 = max(blobs(A)); A2 = L.step(A, n); b1 = max(blobs(A2))
    H, W = A.shape
    dy = (b1[1]-b0[1]+H/2) % H - H/2; dx = (b1[2]-b0[2]+W/2) % W - W/2
    return dy/n, dx/n

def _canon():
    _, pat = load_animal('O2u')
    L = Lenia((96, 96))
    A = place(np.zeros((96, 96)), pat, 48, 48)
    A = L.step(A, 100)
    vy, vx = velocity(L, A)
    ang = np.degrees(np.arctan2(vy, vx))
    best = None
    for sgn in (1, -1):
        b = max(blobs(A))
        Ac = np.roll(A, (48-int(round(b[1])), 48-int(round(b[2]))), (0, 1))[24:72, 24:72]
        R = np.clip(ndimage.rotate(Ac, sgn*ang, reshape=False, order=3), 0, 1)
        B = place(np.zeros((96, 96)), R, 48, 48)
        B = L.step(B, 200)
        if not blobs(B): continue
        wy, wx = velocity(L, B)
        if best is None or wx > best[0]:
            b = max(blobs(B))
            Bc = np.roll(B, (48-int(round(b[1])), 48-int(round(b[2]))), (0, 1))[28:68, 28:68]
            Bc = Bc * (Bc > 1e-4)
            best = (wx, wy, sgn, Bc)
    return best

_c = _canon()
SPEED = float(np.hypot(_c[0], _c[1]))
CANON = _c[3]          # 40x40, heading ~ +x
ROTSGN = None

def glider(angle=0.0, flip=False):
    """Orbium heading `angle` degrees (0=+x, 90=+y i.e. down the screen)."""
    p = CANON[::-1] if flip else CANON
    if angle % 360 == 0: return p
    return np.clip(ndimage.rotate(p, -angle, reshape=False, order=3), 0, 1)

if __name__ == '__main__':
    print('speed', SPEED, 'residual heading', np.degrees(np.arctan2(_c[1], _c[0])))
    L = Lenia((128, 128))
    for ang in (0, 45, 90, 135, 180, 250):
        for fl in (False, True):
            A = place(np.zeros((128, 128)), glider(ang, fl), 64, 64)
            A = L.step(A, 60)
            vy, vx = velocity(L, A)
            print(ang, fl, round(np.degrees(np.arctan2(vy, vx)) % 360, 1), round(np.hypot(vy, vx), 3), round(max(blobs(A))[0], 1))
