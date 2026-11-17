"""Time-max projections of the four XOR input cases (R=26 circuit)."""
import numpy as np, json
from multiprocessing import Pool
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hires import *
import xor_hires as X

def proj(args):
    ray, a, b, T = args
    S = X.build(ray); acc = [np.zeros((X.N, X.N), np.float32), np.zeros((X.N, X.N), np.float32)]
    def cb(t, A):
        acc[0] = np.maximum(acc[0], A); 
        m = A > 0.05; acc[1][m] = t
    A = S.run(T, {'x': bool(a), 'y': bool(b)}, every=3, cb=cb)
    return a, b, acc[0], acc[1], S.g

if __name__ == '__main__':
    ray = json.load(open('xor_hires.json'))['ray']; T = 500
    with Pool(4) as p: res = p.map(proj, [(ray, a, b, T) for a in (0, 1) for b in (0, 1)])
    fig, axs = plt.subplots(2, 2, figsize=(13, 9.6), facecolor='#0d1117')
    x0, x1, y0, y1 = -420, 560, -330, 420
    ox, oy = X.ORIGIN
    for ax, (a, b, mx, tt, g) in zip(axs.flat, res):
        img = plt.cm.viridis(tt/T)[..., :3] * np.clip(mx*1.6, 0, 1)[..., None]
        ax.imshow(img[np.ix_(np.arange(oy+y0, oy+y1) % X.N, np.arange(ox+x0, ox+x1) % X.N)], extent=(x0, x1, y1, y0), interpolation='bilinear')
        for n, (gx, gy, h) in g.items():
            present = not ((n == 'x' and not a) or (n == 'y' and not b))
            ax.annotate(n, (gx, gy), xytext=(gx, gy-34), color='white' if present else '#666', ha='center', fontsize=11,
                        fontweight='bold')
            if not present: ax.plot(gx, gy, 'x', color='#666', ms=9)
        out = a ^ b
        ax.set_title(f'x={a}  y={b}   →   c3 survives: {out}   (x XOR y = {out})', color='white', fontsize=13)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor('black')
    fig.suptitle('XOR from five Orbium gliders in Lenia (R=26): time-max projection, colour = time (dark→yellow)', color='white', fontsize=14)
    plt.tight_layout(); plt.savefig('xor_hires.png', dpi=110, facecolor=fig.get_facecolor())
