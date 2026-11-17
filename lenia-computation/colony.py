"""Growth law of the dividing-ring colony (found_212 index 40)."""
import numpy as np, sys
from mk import *
e = load_found()[40]; N = 768; L = MK((N, N), e['params']); A = L.seed(cells(e))
print(e['params'])
yy, xx = np.mgrid[0:N, 0:N]
for t in range(0, 4201, 600):
    tot = A[0]; bl = blobs(tot, min_mass=5); M = tot.sum()
    rg = np.sqrt((tot*((yy-N/2)**2+(xx-N/2)**2)).sum()/M)
    ms = sorted(round(m) for m, _, _ in bl)
    print(t, 'cells', len(bl), 'mass', round(float(M)), 'radius of gyration', round(float(rg), 1), 'median cell mass', ms[len(ms)//2], flush=True)
    A = L.step(A, 600)
