#!/usr/bin/env python3
"""Rend chaque toml/*.toml du corpus avec fractall-cli.

Sortie:
  bench/fractall/<name>.png
  bench/fractall/_timings.csv  (colonnes: name, status, render_sec, exit_code, stderr_lines)

Usage:
  python3 scripts/render_corpus.py
  python3 scripts/render_corpus.py --width 640 --height 360
  python3 scripts/render_corpus.py --only seahorse,spiral,dragon
  python3 scripts/render_corpus.py --timeout 120
  python3 scripts/render_corpus.py --rebuild   # force cargo build --release avant

Contexte: harness P0 « parité F3 sur corpus toml/ ». Cf. TODO.md.
Le côté F3 est dans scripts/compare_f3.py (bloqué tant que F3 binaire macOS
ne sait pas écrire d'images).
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CLI = REPO / "target" / "release" / "fractall-cli"
TOML_DIR = REPO / "toml"
OUT_DIR = REPO / "bench" / "fractall"


def ensure_cli(rebuild: bool) -> None:
    if rebuild or not CLI.exists():
        print(f"[build] cargo build --release --bin fractall-cli", file=sys.stderr)
        r = subprocess.run(
            ["cargo", "build", "--release", "--bin", "fractall-cli"],
            cwd=REPO,
        )
        if r.returncode != 0:
            sys.exit(f"cargo build failed (exit {r.returncode})")
    if not CLI.exists():
        sys.exit(f"fractall-cli binaire absent après build: {CLI}")


def collect_targets(only: list[str] | None) -> list[Path]:
    files = sorted(TOML_DIR.glob("*.toml"))
    if only:
        wanted = set(only)
        files = [p for p in files if p.stem in wanted]
        missing = wanted - {p.stem for p in files}
        if missing:
            sys.exit(f"TOML inconnus: {sorted(missing)}")
    return files


def render_one(
    toml_path: Path,
    out_path: Path,
    width: int,
    height: int,
    timeout: float,
) -> tuple[str, float, int, list[str]]:
    cmd = [
        str(CLI),
        "--toml", str(toml_path),
        "--width", str(width),
        "--height", str(height),
        "--output", str(out_path),
    ]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
        )
        dt = time.monotonic() - t0
        warns = [
            line for line in (proc.stderr or "").splitlines()
            if "TOML" in line or "WARN" in line.upper() or "non encore appliqué" in line
        ]
        status = "ok" if proc.returncode == 0 and out_path.exists() else "fail"
        return status, dt, proc.returncode, warns
    except subprocess.TimeoutExpired:
        dt = time.monotonic() - t0
        return "timeout", dt, -1, [f"timeout {timeout}s"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=int, default=1024, help="Largeur (défaut F3: 1024)")
    ap.add_argument("--height", type=int, default=576, help="Hauteur (défaut F3: 576)")
    ap.add_argument("--timeout", type=float, default=180.0, help="Timeout par rendu en secondes")
    ap.add_argument("--only", type=str, help="Liste CSV de stems à rendre (ex: seahorse,spiral)")
    ap.add_argument("--rebuild", action="store_true", help="Force cargo build --release")
    ap.add_argument("--clean", action="store_true", help="Vide bench/fractall/ avant")
    args = ap.parse_args()

    if not TOML_DIR.is_dir():
        sys.exit(f"Corpus introuvable: {TOML_DIR}")

    ensure_cli(args.rebuild)

    if args.clean and OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    only = [s.strip() for s in args.only.split(",")] if args.only else None
    targets = collect_targets(only)
    if not targets:
        sys.exit("Aucun TOML à rendre.")

    csv_path = OUT_DIR / "_timings.csv"
    rows: list[dict] = []
    total_start = time.monotonic()
    n = len(targets)
    for i, toml_path in enumerate(targets, 1):
        name = toml_path.stem
        out_path = OUT_DIR / f"{name}.png"
        print(f"[{i:>3}/{n}] {name:30s} ", end="", flush=True)
        status, dt, rc, warns = render_one(toml_path, out_path, args.width, args.height, args.timeout)
        marker = {"ok": "✓", "fail": "✗", "timeout": "⧖"}.get(status, "?")
        print(f"{marker} {status:<8} {dt:6.2f}s  rc={rc}")
        for w in warns:
            print(f"        {w}")
        rows.append({
            "name": name,
            "status": status,
            "render_sec": f"{dt:.3f}",
            "exit_code": rc,
            "stderr_lines": " | ".join(warns),
        })

    total = time.monotonic() - total_start
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "status", "render_sec", "exit_code", "stderr_lines"])
        w.writeheader()
        w.writerows(rows)

    ok = sum(1 for r in rows if r["status"] == "ok")
    print()
    print(f"Terminé en {total:.1f}s — {ok}/{n} OK. CSV: {csv_path.relative_to(REPO)}")
    if ok < n:
        sys.exit(1)


if __name__ == "__main__":
    main()
