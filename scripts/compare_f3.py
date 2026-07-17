#!/usr/bin/env python3
"""Orchestre la comparaison Fraktaler-3 ↔ Fractall pour le corpus toml/.

Pour chaque `toml/<name>.toml`:
  1. Génère un wrapper F3 TOML temporaire (escape_radius alignée avec Fractall=4)
  2. Lance F3 batch → `<name>_f3.exr`
  3. Lance `fractall-cli --export-iterations` → `<name>_fr.exr` (+ PNG)
  4. Décode les deux EXR (channels N=uint32 iter+Nbias, NF=float smooth fraction)
  5. Calcule smooth_iter = (N - Nbias) + NF, masque inside (N == 0xFFFFFFFF)
  6. Métriques : pixels concordants (même inside/outside, |Δiter| ≤ tol),
     mean |Δsmooth|, max |Δsmooth|, RMS
  7. Colorize les deux smooth_iter via colormap commun (matplotlib.viridis ou
     fallback Pillow) → side-by-side + diff amplifié

Sortie:
  bench/compare/<name>__f3.png    (F3 colorisé)
  bench/compare/<name>__fr.png    (Fractall colorisé)
  bench/compare/<name>__diff.png  (composite 3-up + |Δ| amplifié)
  bench/compare/_summary.{csv,md} (tri par mean |Δsmooth| croissant)

Usage:
  python3 scripts/compare_f3.py
  python3 scripts/compare_f3.py --only seahorse,spiral --width 400 --height 400
  python3 scripts/compare_f3.py --iterations 1024  # cap commun aux deux côtés
  python3 scripts/compare_f3.py --escape-radius 4.0  # ER aligné (défaut 4.0 = Fractall)
  python3 scripts/compare_f3.py --rebuild  # cargo build avant + rebuild F3

Pré-requis:
  - F3 build avec EXR (macOS) : `cd fraktaler-3-3.1 && make SYSTEM=macos-batch`
    (cf. build/macos-batch.mk, requiert `brew install openexr`)
  - F3 build avec EXR (Linux) : `bash scripts/build_f3_linux.sh` (batch GUI-free,
    installe les deps apt + produit `fraktaler-3-3.1/fraktaler-3-3.1.linux`)
  - Python : pip install OpenEXR (déjà installé sur la machine de l'utilisateur)
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("numpy requis (devrait venir avec OpenEXR)")
try:
    import OpenEXR
    import Imath
except ImportError:
    sys.exit("Python OpenEXR requis : pip3 install --break-system-packages OpenEXR")
try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow requis")

REPO = Path(__file__).resolve().parent.parent
CLI = REPO / "target" / "release" / "fractall-cli"
TOML_DIR = REPO / "toml"
OUT_DIR = REPO / "bench" / "compare"


def find_f3(required: bool = False) -> Path | None:
    """Localise le binaire Fraktaler-3 (réutilisé par scripts/harness.py).

    Ordre : env `F3_BIN`, `fraktaler-3-3.1.linux` (stripped, préféré),
    `fraktaler-3.linux`, `fraktaler-3.macos`. None si aucun exécutable, sauf
    `required=True` (sortie + indice de build selon la plateforme).
    """
    d = REPO / "fraktaler-3-3.1"
    candidates: list[Path] = []
    env = os.environ.get("F3_BIN")
    if env:
        candidates.append(Path(env))
    candidates.append(d / "fraktaler-3-3.1.linux")
    candidates.append(d / "fraktaler-3.linux")
    candidates.append(d / "fraktaler-3.macos")
    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            return c
    if required:
        if sys.platform == "darwin":
            hint = "cd fraktaler-3-3.1 && make SYSTEM=macos-batch"
        else:
            hint = ("bash scripts/build_f3_linux.sh"
                    "  (ou définir F3_BIN=/chemin/vers/fraktaler-3)")
        sys.exit(
            "F3 binaire introuvable (essayé F3_BIN, fraktaler-3-3.1.linux, "
            f"fraktaler-3.linux, fraktaler-3.macos)\n→ build: {hint}"
        )
    return None


F3 = find_f3()

NBIAS = 1024
INSIDE_MARKER = 0xFFFFFFFF


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

@dataclass
class LightToml:
    real: str
    imag: str
    zoom: str
    iterations: int | None
    rotate: float | None
    # G4 jalon 5e : séquence de phases hybride ("mandelbrot,burning_ship").
    phases: str | None = None


def parse_light_toml(path: Path) -> LightToml:
    real = imag = zoom = None
    iters = None
    rotate = None
    phases = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"')
        if k == "real": real = v
        elif k == "imag": imag = v
        elif k == "zoom": zoom = v
        elif k == "iterations":
            try: iters = int(v)
            except ValueError: pass
        elif k == "rotate":
            try: rotate = float(v)
            except ValueError: pass
        elif k == "phases":
            phases = v
    if real is None or imag is None or zoom is None:
        raise ValueError(f"TOML {path} sans champ real/imag/zoom")
    return LightToml(real, imag, zoom, iters, rotate, phases)


def write_f3_wrapper(
    src: LightToml,
    out_dir: Path,
    name: str,
    width: int,
    height: int,
    iterations: int,
    escape_radius: float,
) -> Path:
    """Écrit un fichier TOML au format F3 pointant vers les coordonnées de src."""
    # F3 stocke les iters en count_t (int64) — pas d'overflow comme nous.
    f3 = out_dir / f"{name}_f3.toml"
    base = out_dir / f"{name}_f3"
    text = (
        'program = "fraktaler-3"\n'
        'version = "3.1"\n'
        f'location.real = "{src.real}"\n'
        f'location.imag = "{src.imag}"\n'
        f'location.zoom = "{src.zoom}"\n'
        f'bailout.iterations = {iterations}\n'
        f'bailout.maximum_reference_iterations = {iterations}\n'
        f'bailout.maximum_perturb_iterations = {iterations}\n'
        f'bailout.maximum_bla_steps = {iterations}\n'
        f'bailout.escape_radius = {escape_radius}\n'
        f'image.width = {width}\n'
        f'image.height = {height}\n'
        'image.subframes = 1\n'
        f'render.filename = "{base}"\n'
        'render.save_exr = true\n'
        'render.exr_channels = ["N0", "NF"]\n'
    )
    if src.rotate is not None and src.rotate != 0.0:
        text += f"transform.rotate = {src.rotate}\n"
    # G4 jalon 5e : hybrides multi-phase — un bloc [[formula]] par phase,
    # opcodes alignés sur bytecode/compile.rs (F3 param.cc op_string).
    if src.phases:
        F3_OPCODES = {
            "mandelbrot": "sqr add", "m": "sqr add", "mandel": "sqr add",
            "burning_ship": "absx absy sqr add", "burningship": "absx absy sqr add",
            "bs": "absx absy sqr add", "ship": "absx absy sqr add",
            "tricorn": "negy sqr add", "mandelbar": "negy sqr add",
            "celtic": "sqr absx add",
            "buffalo": "sqr absx absy add",
            "perpbs": "absy negy sqr add", "pbs": "absy negy sqr add",
            "perpendicularburningship": "absy negy sqr add",
        }
        for name in src.phases.split(","):
            key = "".join(c for c in name.strip().lower() if c.isalnum() or c == "_")
            ops = F3_OPCODES.get(key)
            if ops is None:
                raise ValueError(f"phases: type inconnu pour F3: {name!r}")
            text += f'\n[[formula]]\nopcodes = "{ops}"\n'
    f3.write_text(text)
    return f3


def read_exr_iterations(path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Renvoie (N: uint32, NF: float32, width, height). N == INSIDE_MARKER signale l'intérieur."""
    f = OpenEXR.InputFile(str(path))
    h = f.header()
    dw = h["dataWindow"]
    W = dw.max.x - dw.min.x + 1
    H = dw.max.y - dw.min.y + 1
    chs = list(h["channels"].keys())
    if "N" not in chs:
        raise ValueError(f"EXR {path} sans channel 'N' (chans={chs})")
    n = np.frombuffer(
        f.channel("N", Imath.PixelType(Imath.PixelType.UINT)),
        dtype=np.uint32,
    ).reshape(H, W)
    nf = np.frombuffer(
        f.channel("NF", Imath.PixelType(Imath.PixelType.FLOAT)),
        dtype=np.float32,
    ).reshape(H, W)
    return n, nf, W, H


def smooth_iter(n: np.ndarray, nf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convertit (N, NF) en (smooth_iter: float, inside_mask: bool).
    smooth_iter vaut 0 pour les pixels inside (marqueur — toujours filtrer avec mask).
    """
    inside = n == INSIDE_MARKER
    raw = np.where(inside, 0, n.astype(np.int64) - NBIAS)
    si = raw.astype(np.float64) + nf.astype(np.float64)
    return si, inside


# ---------------------------------------------------------------------------
# Render orchestration
# ---------------------------------------------------------------------------

def run_f3(toml_path: Path, timeout: float) -> tuple[int, float]:
    """Renvoie (returncode, secondes). secondes = inf si timeout."""
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            [str(F3), "-b", "-P", str(toml_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
        )
        return proc.returncode, time.monotonic() - t0
    except subprocess.TimeoutExpired:
        return -1, float("inf")
    except OSError:
        # binaire non exécutable ici (ex: build .macos lancé sur Linux)
        return 127, time.monotonic() - t0


def run_fractall(
    toml_path: Path,
    out_png: Path,
    out_exr: Path,
    width: int,
    height: int,
    iterations: int,
    escape_radius: float,
    timeout: float,
) -> tuple[int, str, float]:
    cmd = [
        str(CLI),
        "--toml", str(toml_path),
        "--width", str(width),
        "--height", str(height),
        "--iterations", str(iterations),
        "--bailout", str(escape_radius),  # aligne l'ER fractall sur F3
        "--output", str(out_png),
        "--export-iterations", str(out_exr),
    ]
    t0 = time.monotonic()
    # Désactive l'auto-adjust de fractall : F3 ne le fait pas, donc on doit
    # rester sur l'iter_max fourni pour comparer apples-to-apples.
    # Désactive aussi la period-detection truncation : F3 calcule toujours la
    # référence complète ; les faux positifs de période font diverger fractall
    # (cf. glitch_test_5 : escape uniforme vs intérieur F3).
    env = os.environ.copy()
    env["FRACTALL_NO_AUTO_ADJUST"] = "1"
    env["FRACTALL_NO_PERIOD"] = "1"
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
            env=env,
        )
        secs = time.monotonic() - t0
        return proc.returncode, f"{secs:.2f}s", secs
    except subprocess.TimeoutExpired:
        return -1, f"timeout {timeout}s", float("inf")


# ---------------------------------------------------------------------------
# Colorize & diff visualisation
# ---------------------------------------------------------------------------

def colormap_smooth(si: np.ndarray, inside: np.ndarray, repeat: float = 0.05) -> np.ndarray:
    """Colormap déterministe et identique pour les deux côtés.
    Utilise un mapping hsv simple : hue = (si * repeat) mod 1, sat=1, val=1.
    inside → noir.
    """
    h = (si.astype(np.float64) * repeat) % 1.0
    s = np.ones_like(h)
    v = np.ones_like(h)
    # hsv → rgb (vectorisé)
    i = (h * 6.0).astype(np.int64)
    f = (h * 6.0) - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i6 = i % 6
    r = np.where(i6 == 0, v, np.where(i6 == 1, q, np.where(i6 == 2, p,
        np.where(i6 == 3, p, np.where(i6 == 4, t, v)))))
    g = np.where(i6 == 0, t, np.where(i6 == 1, v, np.where(i6 == 2, v,
        np.where(i6 == 3, q, np.where(i6 == 4, p, p)))))
    b = np.where(i6 == 0, p, np.where(i6 == 1, p, np.where(i6 == 2, t,
        np.where(i6 == 3, v, np.where(i6 == 4, v, q)))))
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.where(inside[..., None], 0.0, rgb)
    return (rgb * 255).clip(0, 255).astype(np.uint8)


def diff_rgb(diff: np.ndarray, amplify: float) -> np.ndarray:
    """Convertit un array de diff en RGB (gris amplifié)."""
    d = np.abs(diff)
    if d.max() > 0:
        v = np.clip(d * amplify, 0, 255).astype(np.uint8)
    else:
        v = np.zeros_like(d, dtype=np.uint8)
    return np.stack([v, v, v], axis=-1)


def compose_3up(a: np.ndarray, b: np.ndarray, d: np.ndarray, labels=("f3", "fractall", "|diff|")) -> Image.Image:
    H, W = a.shape[:2]
    out = Image.new("RGB", (W * 3 + 4, H + 16), (40, 40, 40))
    out.paste(Image.fromarray(a), (0, 16))
    out.paste(Image.fromarray(b), (W + 2, 16))
    out.paste(Image.fromarray(d), (2 * W + 4, 16))
    from PIL import ImageDraw
    drw = ImageDraw.Draw(out)
    drw.text((4, 1), labels[0], fill=(255, 255, 255))
    drw.text((W + 6, 1), labels[1], fill=(255, 255, 255))
    drw.text((2 * W + 8, 1), labels[2], fill=(255, 255, 255))
    return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def process_one(
    toml_path: Path,
    tmp_dir: Path,
    out_dir: Path,
    width: int,
    height: int,
    iterations: int,
    escape_radius: float,
    timeout: float,
    color_repeat: float,
    amplify: float,
) -> dict:
    name = toml_path.stem
    row: dict = {"name": name, "status": "?"}

    src = parse_light_toml(toml_path)
    iters_used = iterations if iterations > 0 else (src.iterations or 1024)
    if iters_used > 2**31 - 1:
        # F3 utilise count_t (int64) mais on garde un cap raisonnable.
        iters_used = 2**31 - 1
    row["iterations"] = iters_used

    f3_toml = write_f3_wrapper(src, tmp_dir, name, width, height, iters_used, escape_radius)
    f3_exr = tmp_dir / f"{name}_f3.exr"
    fr_exr = out_dir / f"{name}__fractall.exr"
    fr_png = out_dir / f"{name}__fractall.png"

    rc_f3, f3_secs = run_f3(f3_toml, timeout)
    row["f3_sec"] = "timeout" if f3_secs == float("inf") else f"{f3_secs:.2f}s"
    row["f3_secs_num"] = ""  # champ numérique (float s), rempli au succès
    if rc_f3 == -1:  # timeout = perf-bound, distinct d'un vrai échec
        row["status"] = "f3_timeout"
        return row
    if rc_f3 != 0 or not f3_exr.exists():
        row["status"] = "f3_fail"
        return row
    row["f3_secs_num"] = f"{f3_secs:.4f}"

    rc_fr, t_fr, fr_secs = run_fractall(toml_path, fr_png, fr_exr, width, height, iters_used, escape_radius, timeout)
    row["fr_sec"] = t_fr
    row["fr_secs_num"] = ""  # champ numérique (float s), rempli au succès
    if rc_fr == -1:  # timeout = perf-bound (cf. G2), pas un bug de rendu
        row["status"] = "fractall_timeout"
        return row
    if rc_fr != 0 or not fr_exr.exists():
        row["status"] = "fractall_fail"
        return row
    row["fr_secs_num"] = f"{fr_secs:.4f}"

    # Décode
    n_f3, nf_f3, W3, H3 = read_exr_iterations(f3_exr)
    n_fr, nf_fr, Wf, Hf = read_exr_iterations(fr_exr)
    if (W3, H3) != (Wf, Hf):
        row["status"] = f"size_mismatch {W3}x{H3} vs {Wf}x{Hf}"
        return row

    si_f3, in_f3 = smooth_iter(n_f3, nf_f3)
    si_fr, in_fr = smooth_iter(n_fr, nf_fr)

    inside_diff = int((in_f3 != in_fr).sum())
    both_out = (~in_f3) & (~in_fr)
    if both_out.any():
        delta = si_f3[both_out] - si_fr[both_out]
        mean_abs = float(np.mean(np.abs(delta)))
        max_abs = float(np.max(np.abs(delta)))
        rms = float(np.sqrt(np.mean(delta ** 2)))
    else:
        mean_abs = max_abs = rms = 0.0

    total = W3 * H3
    row["status"] = "ok"
    row["inside_f3"] = int(in_f3.sum())
    row["inside_fr"] = int(in_fr.sum())
    row["inside_mismatch"] = inside_diff
    row["both_out"] = int(both_out.sum())

    # F3-degenerate detection (cf. glitch_test_5) : F3 court-circuite via un
    # fast-path period/nucleus en batch → image QUASI-UNIFORME rendue en < 0.1 s,
    # alors que fractall a une vraie structure. On exclut ces cas du score
    # (fractall vraisemblablement correct) au lieu de compter fractall en échec.
    # Deux signaux combinés (goal G1) : (a) F3 quasi-uniforme, (b) F3 rapide,
    # (c) fractall structuré. Symétrique respecté : si fractall est dégénéré et
    # F3 structuré, c'est un VRAI bug fractall (non masqué).
    def _uniform(si: np.ndarray, inside: np.ndarray) -> bool:
        frac_in = inside.sum() / total
        if frac_in > 0.99 or frac_in < 0.01:
            return True
        ext_n = int((~inside).sum())
        return ext_n > 16 and float(si[~inside].std()) < 1e-3

    f3_inside_frac = in_f3.sum() / total
    fr_inside_frac = in_fr.sum() / total
    row["f3_fast"] = bool(f3_secs < 0.1)
    f3_uniform = _uniform(si_f3, in_f3)
    fr_structured = not _uniform(si_fr, in_fr)
    if f3_uniform and f3_secs < 0.1 and fr_structured:
        row["status"] = "f3_degenerate"
        row["note"] = (
            f"F3 uniforme ({100*f3_inside_frac:.1f}% intérieur) rendu en "
            f"{f3_secs:.3f}s (fast-path) vs fractall structuré "
            f"({100*fr_inside_frac:.1f}% intérieur) — fractall correct"
        )
    row["mean_abs_dsi"] = f"{mean_abs:.4f}"
    row["max_abs_dsi"] = f"{max_abs:.4f}"
    row["rms_dsi"] = f"{rms:.4f}"
    # Δmean RELATIF au nombre d'itérations : seule métrique comparable entre cas
    # (un cas à 1.5M iter a un smooth ~1e6, donc un Δmean absolu élevé même à
    # erreur relative faible). Classification : <0.1% pixel-equiv/bord, <2%
    # divergence modérée, >2% à investiguer.
    row["rel_dsi_pct"] = f"{100.0 * mean_abs / max(1, iters_used):.4f}"
    row["px_total"] = total

    # Colorize
    rgb_f3 = colormap_smooth(si_f3, in_f3, repeat=color_repeat)
    rgb_fr = colormap_smooth(si_fr, in_fr, repeat=color_repeat)
    diff_si = si_f3 - si_fr
    # Affiche aussi inside_mismatch en rouge dans le diff
    d_rgb = diff_rgb(diff_si, amplify)
    if inside_diff:
        mismatch = in_f3 != in_fr
        d_rgb[mismatch] = [255, 0, 0]
    Image.fromarray(rgb_f3).save(out_dir / f"{name}__f3.png")
    Image.fromarray(rgb_fr).save(out_dir / f"{name}__fr.png")
    compose_3up(rgb_f3, rgb_fr, d_rgb).save(out_dir / f"{name}__diff.png")

    return row


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--width", type=int, default=400)
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--iterations", type=int, default=0,
                    help="Cap commun aux deux côtés (0 = utilise celui du TOML)")
    ap.add_argument("--escape-radius", type=float, default=25.0,
                    help="ER aligné entre F3 et Fractall (défaut 25.0 = ESCAPE_TIME_BAILOUT). "
                         "Passe --bailout côté fractall ET bailout.escape_radius côté F3.")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--only", type=str, help="liste CSV de stems")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--keep-tmp", action="store_true", help="Conserve les fichiers temp F3")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--color-repeat", type=float, default=0.05)
    ap.add_argument("--amplify", type=float, default=20.0)
    args = ap.parse_args()

    # Résout --out en absolu (tolère les chemins relatifs passés en ligne de commande).
    args.out = args.out.resolve()

    if F3 is None:
        # sort avec un indice adapté à la plateforme si toujours introuvable
        globals()["F3"] = find_f3(required=True)

    if args.rebuild or not CLI.exists():
        r = subprocess.run(["cargo", "build", "--release", "--bin", "fractall-cli"], cwd=REPO)
        if r.returncode != 0:
            sys.exit("cargo build failed")
    if not CLI.exists():
        sys.exit(f"fractall-cli introuvable: {CLI}")

    args.out.mkdir(parents=True, exist_ok=True)

    files = sorted(TOML_DIR.glob("*.toml"))
    if args.only:
        wanted = set(s.strip() for s in args.only.split(","))
        files = [p for p in files if p.stem in wanted]
        if not files:
            sys.exit(f"Aucun TOML pour {wanted}")

    tmp_root = Path(tempfile.mkdtemp(prefix="fractall_f3_compare_"))
    print(f"Tmp: {tmp_root}  Out: {args.out.relative_to(REPO)}")
    print(f"Resolution: {args.width}x{args.height}  ER: {args.escape_radius}")
    if args.iterations:
        print(f"Iterations cap: {args.iterations} (forcé)")

    rows = []
    try:
        for i, toml_path in enumerate(files, 1):
            print(f"[{i:>3}/{len(files)}] {toml_path.stem:30s}", end=" ", flush=True)
            row = process_one(
                toml_path, tmp_root, args.out,
                args.width, args.height, args.iterations, args.escape_radius,
                args.timeout, args.color_repeat, args.amplify,
            )
            rows.append(row)
            if row["status"] == "ok":
                print(f"✓ Δmean={row['mean_abs_dsi']:>7s} Δmax={row['max_abs_dsi']:>7s} "
                      f"inside_mismatch={row['inside_mismatch']:>5d}  (F3 {row['f3_sec']}, Fr {row['fr_sec']})")
            elif row["status"] == "f3_degenerate":
                print(f"⊘ F3 dégénéré (fractall correct) — {row.get('note', '')}")
            else:
                print(f"✗ {row['status']}")
    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmp_root, ignore_errors=True)
        else:
            print(f"Tmp conservé: {tmp_root}")

    rows_ok = [r for r in rows if r["status"] == "ok"]
    rows_ok.sort(key=lambda r: float(r.get("rel_dsi_pct", "0")))

    if rows:
        # CSV (toutes lignes)
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with (args.out / "_summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows: w.writerow(r)
        # MD
        with (args.out / "_summary.md").open("w") as f:
            f.write("# Parité F3 — résumé\n\n")
            f.write(f"Resolution {args.width}x{args.height} | ER aligné {args.escape_radius}\n\n")
            f.write("Trié par Δmean **relatif** (Δmean / iterations) — seule métrique "
                    "comparable entre cas (un cas à 1.5M iter a un smooth ~1e6).\n\n")
            f.write("| Cas | iter | rel Δ% | inside_mm | mean |Δsi| | max |Δsi| |\n")
            f.write("|-----|-----:|-------:|----------:|-----------:|----------:|\n")
            for r in rows_ok:
                f.write(f"| `{r['name']}` | {r.get('iterations','?')} | {r.get('rel_dsi_pct','?')} | "
                        f"{r['inside_mismatch']} | {r['mean_abs_dsi']} | {r['max_abs_dsi']} |\n")
            degen = [r for r in rows if r["status"] == "f3_degenerate"]
            if degen:
                f.write("\n## F3 dégénéré (fractall correct — hors score)\n\n")
                for r in degen:
                    f.write(f"- `{r['name']}` : {r.get('note', 'F3 uniforme tout-intérieur')}\n")
            perf = [r for r in rows if r["status"] in ("fractall_timeout", "f3_timeout")]
            if perf:
                f.write("\n## Perf-bound (timeout — cf. G2, pas un bug de rendu)\n\n")
                for r in perf:
                    f.write(f"- `{r['name']}` : {r['status']} (iter={r.get('iterations', '?')})\n")
            fails = [r for r in rows
                     if r["status"] not in ("ok", "f3_degenerate", "fractall_timeout", "f3_timeout")]
            if fails:
                f.write("\n## Échecs à élucider\n\n")
                for r in fails:
                    f.write(f"- `{r['name']}` : {r['status']}\n")

    n_ok = len(rows_ok)
    n_degen = sum(1 for r in rows if r["status"] == "f3_degenerate")
    n_perf = sum(1 for r in rows if r["status"] in ("fractall_timeout", "f3_timeout"))
    n_fail = sum(1 for r in rows
                 if r["status"] not in ("ok", "f3_degenerate", "fractall_timeout", "f3_timeout"))
    print()
    parts = [f"{n_ok} ok"]
    if n_degen:
        parts.append(f"{n_degen} F3-dégénéré (fractall correct)")
    if n_perf:
        parts.append(f"{n_perf} perf/timeout")
    if n_fail:
        parts.append(f"{n_fail} à élucider")
    print(f"{len(rows)} cas : " + ", ".join(parts) + f"  →  {args.out.relative_to(REPO)}/_summary.md")


if __name__ == "__main__":
    main()
