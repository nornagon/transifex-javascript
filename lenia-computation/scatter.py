"""Calibrate the head-on elastic scatter at R=26 and fit the outgoing rays."""
import numpy as np, sys, json
from hires import *

def fit_rays(tr, t_min):
    """Greedy-link blobs across snapshots after t_min; fit p(t) = p0 + vel*t for each."""
    ts = sorted(t for t in tr if t >= t_min)
    paths = [[(ts[0], x, y)] for x, y, m in tr[ts[0]] if 230 < m < 320]
    for t in ts[1:]:
        for p in paths:
            c = [(np.hypot(x-p[-1][1], y-p[-1][2]), x, y) for x, y, m in tr[t] if 230 < m < 320]
            if c and min(c)[0] < 40: p.append((t, *min(c)[1:]))
    rays = []
    for p in paths:
        a = np.array(p); cx = np.polyfit(a[:, 0], a[:, 1], 1); cy = np.polyfit(a[:, 0], a[:, 2], 1)
        res = np.max(np.hypot(np.polyval(cx, a[:, 0])-a[:, 1], np.polyval(cy, a[:, 0])-a[:, 2]))
        rays.append(dict(p0=[cx[1], cy[1]], vel=[cx[0], cy[0]], heading=float(np.degrees(np.arctan2(cy[0], cx[0])) % 360),
                         speed=float(np.hypot(cx[0], cy[0])), n=len(p), resid=float(res)))
    return rays

if __name__ == '__main__':
    b = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    for X0 in (160, 163, 167):
        S = Scene(1024, (512, 512)); S.add('x', -X0, 0, 0); S.add('y', X0, b, 180)
        tr, _ = S.tracks(560, every=10)
        for r in fit_rays(tr, 330): print(X0, {k: (np.round(v, 3).tolist() if not isinstance(v, int) else v) for k, v in r.items()})
