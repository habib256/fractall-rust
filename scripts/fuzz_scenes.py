#!/usr/bin/env python3
"""Générateur DÉTERMINISTE de scènes fuzz pour l'axe `fuzz` du harness (G8.2).

But : la classe e13 (2 px dd-sensibles) a été trouvée PAR ACCIDENT — les presets
quality figés ne couvrent que des lieux choisis. Cet outil échantillonne des
points frontière ALÉATOIRES (mais reproductibles : seed committée dans le
scorecard) sur les 6 familles Mandelbrot-like du moteur, filtrés pour que la
ground truth GMP soit bien conditionnée (échappement invariant à une
perturbation additive 2⁻⁴⁶·max(|c|,span) de c — sinon le verdict pert-vs-GMP mesurerait le
chaos du lieu, pas le moteur ; cf. frontières hirsutes Celtic/Buffalo, TODO G3).

Méthode par scène : point de départ aléatoire dans la vue pleine → descente de
zoom sur cellules frontière (voisinage 3×3 mixte, hors axes de pliage pour les
familles à abs) avec préférence lisse + gate de stabilité, jusqu'au span cible
(zoom tiré dans 10^[5,11] — GMP per-pixel abordable). Reroll si cul-de-sac.

CLI : fuzz_scenes.py --seed N [--n N] → une scène JSON par ligne.
"""
from __future__ import annotations

import argparse
import json
import random

# (type_id fractall-cli, nom, filtre axes de pliage)
TYPES = [
    (3, "mandelbrot", False),
    (13, "burning-ship", True),
    (14, "tricorn", False),
    (19, "celtic", True),
    (8, "buffalo", True),
    (18, "perpbs", True),
]

BAILOUT_SQR = 625.0  # ER 25, défaut moteur


def step(name, x, y, cx, cy):
    if name == "mandelbrot":
        return x * x - y * y + cx, 2 * x * y + cy
    if name == "burning-ship":
        ax, ay = abs(x), abs(y)
        return ax * ax - ay * ay + cx, 2 * ax * ay + cy
    if name == "tricorn":
        return x * x - y * y + cx, -2 * x * y + cy
    if name == "celtic":
        return abs(x * x - y * y) + cx, 2 * x * y + cy
    if name == "buffalo":
        return abs(x * x - y * y) + cx, abs(2 * x * y) + cy
    if name == "perpbs":
        return x * x - y * y + cx, -2 * x * abs(y) + cy
    raise ValueError(name)


def escape(name, cx, cy, itmax):
    x = y = 0.0
    i = 0
    while i < itmax and x * x + y * y < BAILOUT_SQR:
        x, y = step(name, x, y, cx, cy)
        i += 1
    return i


def escape_julia(zx, zy, seed, itmax):
    """Julia z² + c : le PIXEL est z₀, la graine `seed` est c."""
    x, y = zx, zy
    cx, cy = seed
    i = 0
    while i < itmax and x * x + y * y < BAILOUT_SQR:
        x, y = x * x - y * y + cx, 2 * x * y + cy
        i += 1
    return i


def stability(name, cx, cy, span, itmax, esc=None):
    """Fraction d'un 5×5 échantillonné sur la VUE (±span/2) où l'échappement est
    invariant à un bump additif 2⁻⁴⁶·max(|c|, span) — proxy de conditionnement
    de la ground truth. Bump ADDITIF (pas relatif) : un point exactement sur un
    axe (cy=0, antennes) doit être perturbé aussi."""
    if esc is None:
        esc = lambda px, py: escape(name, px, py, itmax)
    ok = 0
    for j in range(5):
        for i in range(5):
            px = cx + (i / 4.0 - 0.5) * span
            py = cy + (j / 4.0 - 0.5) * span
            bump = 2.0 ** -46 * max(abs(px), abs(py), span)
            if esc(px, py) == esc(px + bump, py + bump):
                ok += 1
    return ok / 25.0


def hunt(rng, name, fold_filter, target_span, itmax, grid=25, esc=None,
         start=(-0.5, 0.0, 4.0)):
    """Descente de zoom depuis la vue pleine ; renvoie (cx, cy) ou None.
    `esc(px, py)` : compte d'échappement du point (défaut : famille `name`,
    plan des paramètres) ; `start` : vue de départ (cx, cy, span)."""
    if esc is None:
        esc = lambda px, py: escape(name, px, py, itmax)
    cx, cy, span = start
    min_ax = 0.1 if fold_filter else 0.0
    while span > target_span:
        iters = [[esc(cx + (i / (grid - 1) - 0.5) * span,
                      cy + (j / (grid - 1) - 0.5) * span)
                  for i in range(grid)] for j in range(grid)]
        cands = []
        for j in range(1, grid - 1):
            for i in range(1, grid - 1):
                neigh = [iters[jj][ii] for jj in (j - 1, j, j + 1) for ii in (i - 1, i, i + 1)]
                fin = [v for v in neigh if v < itmax]
                if not fin or not any(v >= itmax for v in neigh):
                    continue
                px = cx + (i / (grid - 1) - 0.5) * span
                py = cy + (j / (grid - 1) - 0.5) * span
                if span > min_ax and (abs(px) < min_ax or abs(py) < min_ax):
                    continue
                m = sum(fin) / len(fin)
                var = sum((v - m) ** 2 for v in fin) / len(fin)
                cands.append((var, px, py))
        if not cands:
            return None
        # top-5 lisses, mélangés déterministiquement pour diversifier les scènes
        cands.sort(key=lambda t: t[0])
        top = cands[: max(5, len(cands) // 8)]
        rng.shuffle(top)
        best = None
        for var, px, py in top[:5]:
            if stability(name, px, py, span / 8.0, itmax, esc) >= 1.0:
                best = (px, py)
                break
        if best is None:
            return None
        cx, cy = best
        span /= 8.0
    return (cx, cy) if stability(name, cx, cy, target_span, itmax, esc) >= 1.0 else None


def generate(seed: int, n: int) -> list[dict]:
    """n scènes déterministes. Chaque scène a son sous-rng (seed, i) → l'ajout
    d'une scène ne change pas les précédentes."""
    scenes = []
    i = 0
    attempt = 0
    while len(scenes) < n and attempt < n * 12:
        rng = random.Random(f"{seed}:{attempt}")
        attempt += 1
        # cycle sur les types (diversité garantie) ; le rng ne tire que le lieu
        type_id, name, fold = TYPES[attempt % len(TYPES)]
        zoom_exp = rng.uniform(5.0, 11.0)
        zoom = 10.0 ** zoom_exp
        itmax = 2048
        pt = hunt(rng, name, fold, 4.0 / zoom, itmax)
        if pt is None:
            continue
        cx, cy = pt
        scenes.append({
            "name": f"fuzz-{seed}-{len(scenes)}-{name}",
            "type_id": type_id,
            "type": name,
            "center_x": repr(cx),
            "center_y": repr(cy),
            "zoom": f"{zoom:.6e}",
            "iterations": itmax,
        })
        i += 1
    return scenes


# Graines Julia des sondes. La descente suit le bord entre points qui
# s'échappent et points prisonniers : il faut un ensemble de Julia REMPLI
# (graine dans M) — lapin de Douady, basilique, bord du cardioïde, bulbe.
JULIA_SEEDS = [
    (-0.12, 0.74),
    (-1.0, 0.0),
    (-0.5, 0.5),
    (-0.4, 0.6),
    (0.0, 0.64),
    (-0.75, 0.05),
]


def generate_julia(seed: int, n: int) -> list[dict]:
    """n scènes Julia (type 4) déterministes, dans le plan de z : la descente
    part de la vue pleine et suit le BORD de l'ensemble, donc le centre de la
    scène n'est (presque) jamais 0.

    Pourquoi cette famille est à part : la référence de perturbation d'une vue
    Julia part du centre de la vue (Z[0] ≠ 0), et jusqu'au 2026-11-06 toutes
    les boucles rebasaient comme si Z[0] = 0 — 92 à 99 % de pixels faux sur
    toute vue non centrée en 0, invisibles parce que le seul preset Julia
    était centré en 0. Suite distincte de `generate` : les scènes des six
    familles Mandelbrot-like d'une seed donnée ne changent pas."""
    scenes = []
    attempt = 0
    while len(scenes) < n and attempt < n * 12:
        rng = random.Random(f"julia:{seed}:{attempt}")
        jseed = JULIA_SEEDS[attempt % len(JULIA_SEEDS)]
        attempt += 1
        zoom = 10.0 ** rng.uniform(5.0, 11.0)
        itmax = 2048
        esc = lambda px, py, js=jseed: escape_julia(px, py, js, itmax)
        pt = hunt(rng, "julia", False, 4.0 / zoom, itmax, esc=esc,
                  start=(0.0, 0.0, 4.0))
        if pt is None:
            continue
        cx, cy = pt
        scenes.append({
            "name": f"fuzz-{seed}-j{len(scenes)}-julia",
            "type_id": 4,
            "type": "julia",
            "center_x": repr(cx),
            "center_y": repr(cy),
            "zoom": f"{zoom:.6e}",
            "iterations": itmax,
            "julia_re": repr(jseed[0]),
            "julia_im": repr(jseed[1]),
        })
    return scenes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--julia", type=int, default=0,
                    help="nombre de scènes Julia à ajouter (suite distincte)")
    args = ap.parse_args()
    for s in generate(args.seed, args.n) + generate_julia(args.seed, args.julia):
        print(json.dumps(s))


if __name__ == "__main__":
    main()
