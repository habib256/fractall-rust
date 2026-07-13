#!/usr/bin/env python3
"""Chasse un point frontière LISSE (faible sensibilité numérique) pour un
preset QA (src/quality/presets.rs) — types à pliage abs (Celtic/Buffalo/PerpBS)
dont les frontières hirsutes sont incomparables vs GMP (cf. TODO G3 2026-07-13).

À chaque niveau de zoom : scan 33×33, cellules frontière (voisinage 3×3 mixte
inside/outside, hors axes de pliage) scorées par lissité (variance des iters
finis du voisinage) + stabilité (itération d'échappement invariante à une
perturbation relative 2^-46 de c, grille 5×5). Recentre sur la meilleure,
zoome ×8 jusqu'au span cible.

Usage : hunt_qa_scene.py <celtic|buffalo|perpbs> <cx0> <cy0> <span0> [span_cible] [itmax]
Ex.    : hunt_qa_scene.py celtic -0.2 0.35 1.0 4e-9 4096

Valider ensuite la scène : fractall-quality compare (PASS attendu) + contrôle
ground truth stable en précision (--precision-bits 512 → mêmes métriques)."""
import sys

def celtic(cx, cy, itmax, bail=625.0):
    x, y = 0.0, 0.0
    for i in range(itmax):
        if x*x + y*y > bail:
            return i
        u = x*x - y*y
        v = 2.0*x*y
        x = abs(u) + cx
        y = v + cy
    return itmax

def buffalo(cx, cy, itmax, bail=625.0):
    x, y = 0.0, 0.0
    for i in range(itmax):
        if x*x + y*y > bail:
            return i
        u = x*x - y*y
        v = 2.0*x*y
        x = abs(u) + cx
        y = abs(v) + cy
    return itmax

def perpbs(cx, cy, itmax, bail=625.0):
    x, y = 0.0, 0.0
    for i in range(itmax):
        if x*x + y*y > bail:
            return i
        nx = x*x - y*y + cx
        ny = -2.0*x*abs(y) + cy
        x, y = nx, ny
    return itmax

FN = {"celtic": celtic, "buffalo": buffalo, "perpbs": perpbs}

def stability(fn, cx, cy, itmax):
    """Fraction de 5×5 px (pas 2^-48·|c|) où iter est invariant au bump 2^-46."""
    eps = 2.0**-46
    ok = tot = 0
    for j in range(5):
        for i in range(5):
            px = cx * (1.0 + (i-2)*eps)
            py = cy * (1.0 + (j-2)*eps)
            a = fn(px, py, itmax)
            b = fn(px*(1.0+eps), py*(1.0+eps), itmax)
            tot += 1
            if a == b:
                ok += 1
    return ok / tot

def hunt(name, cx0, cy0, span0, target_span, itmax, min_ax=0.1):
    fn = FN[name]
    cx, cy, span = cx0, cy0, span0
    n = 33
    while span > target_span:
        iters = [[fn(cx + (i/(n-1)-0.5)*span, cy + (j/(n-1)-0.5)*span, itmax)
                  for i in range(n)] for j in range(n)]
        cands = []
        for j in range(1, n-1):
            for i in range(1, n-1):
                neigh = [iters[jj][ii] for jj in (j-1, j, j+1) for ii in (i-1, i, i+1)]
                fin = [v for v in neigh if v < itmax]
                if not fin or not any(v >= itmax for v in neigh):
                    continue  # pas frontière
                px = cx + (i/(n-1)-0.5)*span
                py = cy + (j/(n-1)-0.5)*span
                if abs(px) < min_ax or abs(py) < min_ax:
                    continue
                m = sum(fin)/len(fin)
                var = sum((v-m)**2 for v in fin)/len(fin)
                cands.append((var, j*n+i, px, py))
        if not cands:
            print(f"[{name}] DEAD END span={span:.3e} ({cx!r},{cy!r})")
            return None
        cands.sort()
        # stabilité sur le top 5 lisses
        best = None
        for var, _, px, py in cands[:5]:
            s = stability(fn, px, py, itmax)
            if best is None or s > best[0] or (s == best[0] and var < best[1]):
                best = (s, var, px, py)
        _, _, cx, cy = best
        span /= 8.0
        print(f"[{name}] span={span:.3e} c=({cx!r},{cy!r}) stab={best[0]:.2f} var={best[1]:.1f}")
    s_final = stability(fn, cx, cy, itmax)
    print(f"FINAL {name}: center=({cx!r}, {cy!r}) stab={s_final:.2f}")
    return cx, cy

if __name__ == "__main__":
    name = sys.argv[1]
    cx0, cy0, s0 = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    target = float(sys.argv[5]) if len(sys.argv) > 5 else 4e-9
    itmax = int(sys.argv[6]) if len(sys.argv) > 6 else 4096
    hunt(name, cx0, cy0, s0, target, itmax)
