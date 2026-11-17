"""Scan 'found' multi-kernel patterns for sustained, sparse growth in creature count."""
import numpy as np, sys, json, re
from multiprocessing import Pool
from mk import *

def run(a):
    path, idx, N, T = a
    e = load_found(path)[idx]
    try:
        L = MK((N, N), e['params']); A = L.seed(cells(e))
    except Exception as ex:
        return (path, idx, e['name'], 'ERR '+str(ex)[:60], [])
    hist = []
    for t in range(0, T+1, T//6):
        tot = A.sum(0); bl = [m for m, _, _ in blobs(tot, min_mass=5)]
        hist.append((t, len(bl), round(float(tot.sum())), round(float((tot > 0.1).mean()), 3)))
        if hist[-1][3] > 0.25 or hist[-1][2] < 1: break
        A = L.step(A, T//6)
    return (path, idx, e['code']+'|'+e['name'], e['params'][0]['R'], hist)

if __name__ == '__main__':
    paths = sys.argv[1:]; jobs = []
    for p in paths:
        for i, e in enumerate(load_found(p)): jobs.append((p, i, 384, 2400))
    res = []
    with Pool(4) as pool:
        for r in pool.imap(run, jobs, chunksize=1):
            res.append(r); path, idx, name, R, hist = r
            if hist:
                ns = [h[1] for h in hist]; last = hist[-1]
                tag = 'DEAD' if last[2] < 1 else 'FILL' if last[3] > 0.25 else ('GROW' if ns[-1] >= ns[1]+3 and ns[-1] > ns[len(ns)//2] else '')
                print(path[-9:], idx, name[:28], 'R', R, 'n', ns, 'mass', [h[2] for h in hist][-3:], 'area', last[3], tag, flush=True)
            else: print(path[-9:], idx, name, R, flush=True)
    json.dump(res, open('mkscan_more.json', 'w')); sys.exit()
    json.dump(res, open('mkscan_'+'_'.join(re.sub(r'\D', '', p.split('/')[-1]) for p in paths)+'.json', 'w'))
    for path, idx, name, R, hist in res:
        if not hist: print(path[-8:], idx, name, R); continue
        ns = [h[1] for h in hist]; last = hist[-1]
        tag = 'DEAD' if last[2] < 1 else 'FILL' if last[3] > 0.25 else ('GROW' if ns[-1] >= ns[1]+3 and ns[-1] > ns[len(ns)//2] else '')
        print(path[-8:], idx, name[:28], 'R', R, 'n', ns, 'mass', [h[2] for h in hist][-3:], 'area', last[3], tag)
