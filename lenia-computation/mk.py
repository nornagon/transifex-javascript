"""Multi-kernel, single-channel Lenia (the '2D, 1 channel, K kernels' family of Chan's
LeniaNDKC.py):  A <- clip(A + dt * sum_k h_k G_k(K_k * A) / sum_k h_k, 0, 1)."""
import numpy as np, re, json, os
from fractions import Fraction
from lenia import rle2arr, place, blobs

HERE = os.path.dirname(os.path.abspath(__file__))

def load_found(path=os.path.join(HERE, 'found_212.json')):
    """The 'found' files are comma-separated JSON values, not one document."""
    s = open(path, encoding='utf-8').read()
    out = []
    for m in re.finditer(r'\{"code":"([^"]*)","name":"([^"]*)","cname":"([^"]*)","params":(\[.*?\]),"cells":("[^"]*"|\[[^\]]*\])', s):
        out.append(dict(code=m.group(1), name=m.group(2), cname=m.group(3), params=json.loads(m.group(4)), cells=json.loads(m.group(5))))
    return out

def shell_kernel(shape, R, p):
    H, W = shape
    y = np.fft.fftfreq(H)*H; x = np.fft.fftfreq(W)*W
    D = np.sqrt(y[:, None]**2 + x[None, :]**2) / R
    r = p.get('r', 1); bs = [float(Fraction(f)) for f in (p['b'].split(',') if isinstance(p['b'], str) else p['b'])]
    B = len(bs); Br = B*D/r
    b = np.asarray(bs)[np.minimum(np.floor(Br).astype(int), B-1)]
    q = np.minimum(Br % 1, 1)
    with np.errstate(divide='ignore', over='ignore'):
        core = np.where((q > 0) & (q < 1), np.exp(4 - 1/np.maximum(q*(1-q), 1e-12)), 0)
    K = (D < r)*core*b
    return np.fft.rfft2(K/K.sum())

def cells(entry):
    """-> array (C, h, w). Channels are separated by '%' in the RLE."""
    raw = entry['cells'] if isinstance(entry['cells'], list) else entry['cells'].rstrip('!').split('%')
    parts = [rle2arr(p.rstrip('!%')) for p in raw]
    h = max(p.shape[0] for p in parts); w = max(p.shape[1] for p in parts)
    out = np.zeros((len(parts), h, w))
    for c, p in enumerate(parts): out[c, :p.shape[0], :p.shape[1]] = p
    return out

class MK:
    """State A has shape (C, H, W). Kernel k reads channel c0 and writes channel c1."""
    def __init__(self, shape, params, scale=1):
        self.shape = shape; self.p = params; self.dt = 1/max(p['T'] for p in params)   # some files carry T=1 on kernel 0 only
        self.Kf = [shell_kernel(shape, params[0]['R']*scale, p) for p in params]
        self.hs = [p.get('h', 1) for p in params]
        self.cc = [tuple(p.get('c', [0, 0])) for p in params]
        self.C = 1 + max(max(c) for c in self.cc)
        self.Hn = [sum(h for h, c in zip(self.hs, self.cc) if c[1] == i) for i in range(self.C)]
    def blank(self): return np.zeros((self.C,)+tuple(self.shape))
    def seed(self, pat, cy=None, cx=None):
        A = self.blank(); H, W = self.shape; h, w = pat.shape[-2:]
        cy = H//2 if cy is None else cy; cx = W//2 if cx is None else cx
        A[:pat.shape[0], cy-h//2:cy-h//2+h, cx-w//2:cx-w//2+w] = pat
        return A
    def step(self, A, n=1):
        for _ in range(n):
            F = np.fft.rfft2(A); D = np.zeros_like(A)
            for Kf, p, h, (c0, c1) in zip(self.Kf, self.p, self.hs, self.cc):
                U = np.fft.irfft2(F[c0]*Kf, s=self.shape)
                D[c1] += h*(2*np.exp(-(U-p['m'])**2/(2*p['s']**2)) - 1)
            for c in range(self.C):
                if self.Hn[c] > 0: A[c] = np.clip(A[c] + self.dt*D[c]/self.Hn[c], 0, 1)
        return A

if __name__ == '__main__':
    F = load_found()
    for idx, e in enumerate(F):
        if e['name'] != 'replicator': continue
        N = 512; L = MK((N, N), e['params']); A = L.seed(cells(e))
        print(idx, e['name'], e['params'])
        for t in range(0, 1501, 300):
            bl = sorted((round(m) for m, _, _ in blobs(A.sum(0), min_mass=5)), reverse=True)
            print('  ', t, 'blobs', len(bl), 'mass', round(float(A.sum())), bl[:12])
            A = L.step(A, 300)
