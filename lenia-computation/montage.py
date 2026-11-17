"""Montage of selected found patterns over time: python3 montage.py out.png file:idx ..."""
import sys, numpy as np
from multiprocessing import Pool
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mk import *
N = 512; TS = (600, 1200, 2000, 3000)
def run(spec):
    f, i = spec.rsplit(':', 1); e = load_found(f)[int(i)]; L = MK((N, N), e['params']); A = L.seed(cells(e)); fr = []; t = 0
    for T in TS:
        A = L.step(A, T-t); t = T; tot = A.sum(0)
        fr.append((np.clip(np.moveaxis(A[:3], 0, -1), 0, 1) if A.shape[0] == 3 else tot, len(blobs(tot, min_mass=5))))
    return spec, fr
if __name__ == '__main__':
    out, specs = sys.argv[1], sys.argv[2:]
    with Pool(4) as p: res = p.map(run, specs)
    fig, axs = plt.subplots(len(specs), len(TS), figsize=(5*len(TS), 5*len(specs)), squeeze=False)
    for r, (spec, fr) in enumerate(res):
        for c, (img, n) in enumerate(fr):
            axs[r, c].imshow(img, cmap='magma', vmin=0, vmax=1); axs[r, c].set_title(f'{spec[-14:]} t={TS[c]} n={n}', fontsize=9); axs[r, c].axis('off')
    plt.tight_layout(); plt.savefig(out, dpi=50)
