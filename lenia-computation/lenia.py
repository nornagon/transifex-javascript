"""Minimal batched Lenia (single kernel, single channel) on a torus, FFT-based.

Standard Lenia (Chan 2019):  A <- clip(A + dt * G(K * A), 0, 1)
  K: ring kernel, shell exp(4 - 1/(r(1-r))), normalised to sum 1
  G(u) = 2*exp(-(u-m)^2 / (2 s^2)) - 1
"""
import json, os, re
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ANIMALS = os.environ.get("LENIA_ANIMALS", os.path.join(HERE, "animals.json"))


def rle2arr(st):
    rows, row, num = [], [], ""
    for tok in re.findall(r"\d+|[p-y]?[.A-Xa-zA-Z]|\$|!", st.rstrip("!") + "$"):
        if tok.isdigit():
            num = tok
            continue
        n = int(num) if num else 1
        num = ""
        if tok == "$":
            rows.append(row)
            rows.extend([[] for _ in range(n - 1)])
            row = []
        else:
            if tok in ".b":
                v = 0
            elif tok == "o":
                v = 255
            elif len(tok) == 1:
                v = ord(tok) - ord("A") + 1
            else:
                v = (ord(tok[0]) - ord("p")) * 24 + ord(tok[1]) - ord("A") + 25
            row.extend([v] * n)
    w = max(len(r) for r in rows)
    a = np.zeros((len(rows), w))
    for i, r in enumerate(rows):
        a[i, : len(r)] = r
    return a / 255.0


def load_animal(code):
    for x in json.load(open(ANIMALS, encoding="utf-8")):
        if x.get("code") == code and "cells" in x:
            return x["params"], rle2arr(x["cells"])
    raise KeyError(code)


def kernel_fft(shape, R):
    H, W = shape
    y = np.fft.fftfreq(H) * H
    x = np.fft.fftfreq(W) * W
    r = np.sqrt(y[:, None] ** 2 + x[None, :] ** 2) / R
    k = np.zeros_like(r)
    m = (r > 0) & (r < 1)
    k[m] = np.exp(4 - 1 / (r[m] * (1 - r[m])))
    k /= k.sum()
    return np.fft.rfft2(k)


class Lenia:
    def __init__(self, shape, R=13, T=10, m=0.15, s=0.015):
        self.shape, self.R, self.T, self.m, self.s = shape, R, T, m, s
        self.Kf = kernel_fft(shape, R)

    def potential(self, A):
        return np.fft.irfft2(np.fft.rfft2(A) * self.Kf, s=self.shape)

    def step(self, A, n=1):
        for _ in range(n):
            U = self.potential(A)
            G = 2 * np.exp(-((U - self.m) ** 2) / (2 * self.s**2)) - 1
            A = np.clip(A + G / self.T, 0, 1)
        return A


def place(A, pat, cy, cx, angle=0.0, flip=False):
    """Add pattern centred at (cy,cx), rotated by `angle` degrees (CCW on screen, y down)."""
    p = pat[:, ::-1] if flip else pat
    if angle % 360:
        p = ndimage.rotate(p, angle, reshape=True, order=1)
    p = np.clip(p, 0, 1)
    h, w = p.shape
    ys = (np.arange(h) + int(round(cy)) - h // 2) % A.shape[-2]
    xs = (np.arange(w) + int(round(cx)) - w // 2) % A.shape[-1]
    A[..., ys[:, None], xs[None, :]] = np.maximum(A[..., ys[:, None], xs[None, :]], p)
    return A


def blobs(A, thresh=0.1, min_mass=5.0):
    """Connected components on a torus -> list of (mass, cy, cx)."""
    H, W = A.shape
    lab, n = ndimage.label(A > thresh)
    # merge labels across wrap
    par = list(range(n + 1))
    def f(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    for a, b in list(zip(lab[0], lab[-1])) + list(zip(lab[:, 0], lab[:, -1])):
        if a and b:
            par[f(a)] = f(b)
    out = []
    roots = {}
    for i in range(1, n + 1):
        roots.setdefault(f(i), []).append(i)
    yy, xx = np.mgrid[0:H, 0:W]
    for ids in roots.values():
        msk = np.isin(lab, ids)
        w = A * msk
        M = w.sum()
        if M < min_mass:
            continue
        # circular mean
        ay = np.angle((w * np.exp(2j * np.pi * yy / H)).sum()) % (2 * np.pi)
        ax = np.angle((w * np.exp(2j * np.pi * xx / W)).sum()) % (2 * np.pi)
        out.append((M, ay * H / (2 * np.pi), ax * W / (2 * np.pi)))
    return out
