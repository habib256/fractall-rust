#!/usr/bin/env python3
"""Three-way accuracy arbiter: fractall-pert vs Fraktaler-3 vs GMP ground truth.

Motivation
----------
The harness `parity` axis compares fractall to Fraktaler-3, but F3 is NOT ground
truth at *moderate* zoom: its wisdom auto-selects a low-precision numeric tier
(e.g. `float`, 24-bit mantissa) where the depth allows, trading exactness for
speed. So a fractall↔F3 divergence does not tell us *who is wrong*. This tool
renders BOTH against pure per-pixel GMP (fractall `--algorithm gmp`, the same
ground truth the quality suite uses) and counts wrong integer escape-counts for
each — the honest correctness arbiter.

Reuses `compare_f3.py` (F3 discovery, F3 toml wrapper, EXR decode, NBIAS).

Usage
-----
  python3 scripts/three_way_gmp.py                     # default Mandelbrot set
  python3 scripts/three_way_gmp.py --width 128         # faster
  python3 scripts/three_way_gmp.py --scene NAME:RE:IM:ZOOM:ITERS  # custom (repeatable)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_f3 as c  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CLI = REPO / "target" / "release" / "fractall-cli"
ESCAPE_RADIUS = 625.0  # matches fractall default bailout 25 (escape_radius = 25²)

# Moderate-zoom Mandelbrot scenes (from src/quality/presets.rs) where F3's tier
# selection is expected to bite. Deep scenes (>1e15) are omitted: there F3 uses
# high-precision tiers and matches fractall (see quick-parity, all mean_abs≈0).
DEFAULT_SCENES = [
    # name, real, imag, zoom, iterations
    ("seahorse-1e8", "-0.743643887037158704752191506114774",
     "0.131825904205311970493132056385139", "1e8", 4096),
    ("misiurewicz-1e12", "-0.77568377", "0.13646737", "1e12", 8192),
    ("mandelbrot-e13", "-1.7499537683537087", "0.0", "1e13", 16384),
    ("mandelbrot-e17", "-1.7499537683537087215208540815925", "0.0", "1e17", 32768),
]


def render_fractall(real, imag, zoom, iters, algo, out_exr, width, height):
    toml = out_exr.with_suffix(".toml")
    toml.write_text(f'real = "{real}"\nimag = "{imag}"\nzoom = "{zoom}"\niterations = {iters}\n')
    env = os.environ.copy()
    env["FRACTALL_NO_AUTO_ADJUST"] = "1"
    env["FRACTALL_NO_PERIOD"] = "1"
    cmd = [
        str(CLI), "--toml", str(toml), "--width", str(width), "--height", str(height),
        "--iterations", str(iters), "--bailout", str(ESCAPE_RADIUS), "--algorithm", algo,
        "--output", str(out_exr.with_suffix(".png")), "--export-iterations", str(out_exr),
    ]
    subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def render_f3(real, imag, zoom, iters, out_dir, name, width, height):
    src = c.LightToml(real, imag, zoom, iters, None)
    toml = c.write_f3_wrapper(src, out_dir, name, width, height, iters, ESCAPE_RADIUS)
    subprocess.run([str(c.F3), "-b", "-P", str(toml)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out_dir / f"{name}_f3.exr"


def escape_counts(exr_path):
    """Integer escape count per pixel (-1 = inside), from an F3-format EXR."""
    n, nf, w, h = c.read_exr_iterations(exr_path)
    inside = n == c.INSIDE_MARKER
    esc = np.where(inside, -1, n.astype(np.int64) - c.NBIAS)
    return esc, inside


def compare(name, real, imag, zoom, iters, out_dir, width, height):
    pert_exr = out_dir / f"{name}_pert.exr"
    gmp_exr = out_dir / f"{name}_gmp.exr"
    render_fractall(real, imag, zoom, iters, "perturbation", pert_exr, width, height)
    render_fractall(real, imag, zoom, iters, "gmp", gmp_exr, width, height)
    f3_exr = render_f3(real, imag, zoom, iters, out_dir, name, width, height)

    ep, ip = escape_counts(pert_exr)
    eg, ig = escape_counts(gmp_exr)
    ef, iff = escape_counts(f3_exr)
    total = ep.size

    def stats(e, ins):
        both = (~ins) & (~ig)
        d = np.abs(e - eg)
        # `wrong` = any nonzero integer-escape diff; `big` = off by >5 (strips the
        # cross-implementation ±1-2 smooth-rounding noise → genuine large errors).
        return {
            "wrong": int((d[both] > 0).sum()), "big": int((d[both] > 5).sum()),
            "mean": float(d[both].mean()) if both.any() else 0.0,
            "max": int(d[both].max()) if both.any() else 0,
            "inside_mm": int((ins != ig).sum()),
        }

    return name, iters, stats(ep, ip), stats(ef, iff), total


def main():
    ap = argparse.ArgumentParser(description="fractall vs F3 vs GMP accuracy arbiter")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--scene", action="append", default=[],
                    help="custom scene NAME:RE:IM:ZOOM:ITERS (repeatable)")
    ap.add_argument("--out", type=Path, default=REPO / "bench" / "three_way")
    args = ap.parse_args()

    if c.F3 is None:
        sys.exit("F3 binaire introuvable — bash scripts/build_f3_linux.sh")
    if not CLI.exists():
        sys.exit("fractall-cli absent — cargo build --release")

    scenes = list(DEFAULT_SCENES)
    for s in args.scene:
        parts = s.split(":")
        if len(parts) != 5:
            sys.exit(f"--scene mal formé: {s!r} (attendu NAME:RE:IM:ZOOM:ITERS)")
        scenes.append((parts[0], parts[1], parts[2], parts[3], int(parts[4])))

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"# Three-way accuracy vs GMP ground truth ({args.width}x{args.height}, ER {ESCAPE_RADIUS})\n")
    hdr = f"{'scene':<18}{'iters':>7} | {'fractall wrong':>22} | {'F3 wrong':>22} | verdict"
    print(hdr); print("-" * len(hdr))
    rows = []
    for name, real, imag, zoom, iters in scenes:
        try:
            r = compare(name, real, imag, zoom, iters, args.out, args.width, args.height)
        except subprocess.CalledProcessError as e:
            print(f"{name:<18}{iters:>7} | RENDER FAILED ({e})")
            continue
        _, it, fp, f3, total = r
        rows.append(r)
        fr_s = f"{fp['big']:>5} (>5) max{fp['max']:>4}"
        f3_s = f"{f3['big']:>5} (>5) max{f3['max']:>4}"
        # Verdict on the >5 "large error" count (mapping-noise-free).
        if f3["big"] > 4 * max(fp["big"], 1):
            verd = f"fractall {f3['big']/max(fp['big'],1):.0f}x more accurate"
        elif fp["big"] > 4 * max(f3["big"], 1):
            verd = f"F3 {fp['big']/max(f3['big'],1):.0f}x more accurate"
        else:
            verd = "comparable"
        print(f"{name:<18}{it:>7} | {fr_s:>18} | {f3_s:>18} | {verd}")
    print("\n(large-error = integer escape-count off by >5 vs GMP; F3 default wisdom")
    print(" picks nt_float 24-bit where float is exponent-viable ⇒ imprecise at moderate")
    print(" zoom. fractall floors at f64 53-bit. Deep zoom >~1e38: float non-viable ⇒ parity.)")
    return rows


if __name__ == "__main__":
    main()
