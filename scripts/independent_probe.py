#!/usr/bin/env python3
"""Sonde ground-truth INDÉPENDANTE (mpmath, pur Python — zéro GMP/MPFR/rug).

But (TODO §G8.2) : dé-corréler un éventuel bug common-mode de notre juge
`gmp.rs` — le rendu « GMP pur » de fractall-quality et le moteur perturbation
partagent rug/GMP et le même code de mapping ; cette sonde ré-implémente les
deux indépendamment (mpmath.mpf + mapping répliqué de escape_time.rs:538-577)
et arbitre pixel par pixel.

Sémantique répliquée EXACTEMENT (src/render/escape_time.rs + src/fractal/gmp.rs) :
  c = center + span · ((idx + 0.5)/dim − 0.5)   (x et y, y écran vers le bas)
  while i < iter_max && |z|² < bailout² : z = f(z) + c ; i += 1
Formules : mandelbrot, julia, burning-ship, tricorn, celtic, buffalo, perpbs
(mêmes définitions que gmp.rs / bytecode/compile.rs).

Usage :
  # Arbitrer les top-divergents d'un rapport quality (params lus dans gmp.png)
  independent_probe.py --scene quality-reports/mandelbrot-e13 [--top 5] [--prec-bits N]
  # Sonder des pixels explicites
  independent_probe.py --scene <dir> --pixel 131,118 --pixel 131,137
Par défaut prec = 2× le precision_bits des métadonnées (contrôle de convergence :
relancer avec --prec-bits 4× doit donner les mêmes comptes).
"""
import argparse
import json
import struct
import sys
import zlib
from pathlib import Path

from mpmath import mp, mpf

# ---------------------------------------------------------------- PNG metadata


def read_fractall_params(png_path: Path) -> dict:
    """Extrait le JSON du chunk tEXt/zTXt clé `fractall-params` (io/png.rs)."""
    data = png_path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"pas un PNG : {png_path}")
    off = 8
    while off < len(data):
        (length,) = struct.unpack(">I", data[off : off + 4])
        ctype = data[off + 4 : off + 8]
        chunk = data[off + 8 : off + 8 + length]
        if ctype == b"tEXt":
            key, _, value = chunk.partition(b"\x00")
            if key == b"fractall-params":
                return json.loads(value.decode("latin-1"))
        elif ctype == b"zTXt":
            key, _, rest = chunk.partition(b"\x00")
            if key == b"fractall-params":
                return json.loads(zlib.decompress(rest[1:]).decode("latin-1"))
        off += 12 + length
    raise ValueError(f"chunk fractall-params absent de {png_path}")


# ---------------------------------------------------------------- itérations
# (x, y) réels séparés — mêmes formules que gmp.rs (abs sur les composantes).


def step_mandelbrot(x, y, cx, cy):
    return x * x - y * y + cx, 2 * x * y + cy


def step_julia(x, y, cx, cy):  # (cx,cy) = seed, injecté par le wrapper
    return x * x - y * y + cx, 2 * x * y + cy


def step_burning_ship(x, y, cx, cy):
    ax, ay = abs(x), abs(y)
    return ax * ax - ay * ay + cx, 2 * ax * ay + cy


def step_tricorn(x, y, cx, cy):
    return x * x - y * y + cx, -2 * x * y + cy


def step_celtic(x, y, cx, cy):
    return abs(x * x - y * y) + cx, 2 * x * y + cy


def step_buffalo(x, y, cx, cy):
    return abs(x * x - y * y) + cx, abs(2 * x * y) + cy


def step_perpbs(x, y, cx, cy):
    return x * x - y * y + cx, -2 * x * abs(y) + cy


STEPS = {
    "Mandelbrot": step_mandelbrot,
    "Julia": step_julia,
    "BurningShip": step_burning_ship,
    "Tricorn": step_tricorn,
    "Celtic": step_celtic,
    "Buffalo": step_buffalo,
    "PerpendicularBurningShip": step_perpbs,
}


def escape_count(ftype, cx, cy, seed, iter_max, bailout_sqr):
    """Réplique mandelbrot_mpc & co : test AVANT le pas, compte des pas faits."""
    step = STEPS[ftype]
    if ftype == "Julia":
        x, y = cx, cy
        px, py = seed
    else:
        x, y = seed
        px, py = cx, cy
    i = 0
    while i < iter_max and x * x + y * y < bailout_sqr:
        x, y = step(x, y, px, py)
        i += 1
    return i


# ---------------------------------------------------------------- mapping


def pixel_to_c(params, px, py):
    """Réplique escape_time.rs:538-577 (AA offset nul, sans rotation)."""
    prec = mp.prec  # déjà positionné par main()
    cx = mpf(params.get("center_x_hp") or repr(params["center_x"]))
    cy = mpf(params.get("center_y_hp") or repr(params["center_y"]))
    sx = mpf(params.get("span_x_hp") or repr(params["span_x"]))
    sy = mpf(params.get("span_y_hp") or repr(params["span_y"]))
    w, h = params["width"], params["height"]
    if params.get("rotation") or params.get("transform_k"):
        raise SystemExit("rotation/transform_k non supportés par la sonde (ajouter si besoin)")
    x_ratio = (mpf(px) + mpf("0.5")) / w - mpf("0.5")
    y_ratio = (mpf(py) + mpf("0.5")) / h - mpf("0.5")
    return cx + sx * x_ratio, cy + sy * y_ratio


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="dossier quality-reports/<preset> (gmp.png + report.json)")
    ap.add_argument("--pixel", action="append", default=[], help="x,y explicite (répétable)")
    ap.add_argument("--top", type=int, default=5, help="sonder les N premiers top-divergents du report.json")
    ap.add_argument("--prec-bits", type=int, default=0, help="précision mpmath en bits (défaut : 2× precision_bits des métadonnées)")
    args = ap.parse_args()

    scene = Path(args.scene)
    params = read_fractall_params(scene / "gmp.png")
    ftype = params["fractal_type"]
    if ftype not in STEPS:
        raise SystemExit(f"type {ftype} non supporté (sonde : {sorted(STEPS)})")

    report = None
    rj = scene / "report.json"
    if rj.exists():
        report = json.loads(rj.read_text())

    prec = args.prec_bits or 2 * int(params.get("precision_bits", 256))
    mp.prec = prec

    iter_max = params["iteration_max"]
    bailout = mpf(repr(params.get("bailout", 25.0)))
    bailout_sqr = bailout * bailout
    raw_seed = params["seed"]  # Complex64 serde : {re,im} ou [re,im] selon version
    if isinstance(raw_seed, dict):
        seed = (mpf(repr(raw_seed["re"])), mpf(repr(raw_seed["im"])))
    else:
        seed = (mpf(repr(raw_seed[0])), mpf(repr(raw_seed[1])))

    pixels = []
    for p in args.pixel:
        x, y = p.split(",")
        pixels.append((int(x), int(y), None, None))
    if not pixels and report:
        for d in (report.get("top_divergent") or [])[: args.top]:
            pixels.append((d["x"], d["y"], d["pert_iter"], d["gmp_iter"]))
    if not pixels:
        raise SystemExit("aucun pixel à sonder (--pixel, ou report.json avec top_divergent)")

    print(f"scene={scene.name} type={ftype} iter_max={iter_max} bailout²={float(bailout_sqr)}")
    print(f"prec mpmath = {prec} b (métadonnées precision_bits={params.get('precision_bits')})")
    print(f"{'pixel':>12} {'mpmath':>9} {'gmp':>9} {'pert':>9}  verdict")
    disagree = 0
    for px, py, pert_it, gmp_it in pixels:
        cx, cy = pixel_to_c(params, px, py)
        n = escape_count(ftype, cx, cy, seed, iter_max, bailout_sqr)
        if gmp_it is None:
            print(f"{px:>5},{py:>6} {n:>9} {'-':>9} {'-':>9}  (pas de rapport)")
            continue
        if n == gmp_it:
            verdict = "mpmath == gmp (juge confirmé)"
        elif n == pert_it:
            verdict = "mpmath == pert (JUGE GMP SUSPECT ⚠️)"
            disagree += 1
        else:
            verdict = "mpmath ≠ les deux (précision ? relancer --prec-bits 2×)"
            disagree += 1
        print(f"{px:>5},{py:>6} {n:>9} {gmp_it:>9} {pert_it:>9}  {verdict}")
    sys.exit(1 if disagree else 0)


if __name__ == "__main__":
    main()
