"""`harness.py hotspots` — profil ÉCHANTILLONNÉ d'un cas (quelles fonctions).

Le profil par phase (`axis_profile`) dit où va le temps ENTRE les phases
(orbite / BLA / pixels / post). Il ne voit pas ce qui se passe DANS une phase :
le 2026-11-06 la phase « pixels » du tier exp cachait un parse GMP par pixel
(70 % du temps), celle des vues peu chères un clone d'`Arc` par pixel, et le
path f64 standard une formule recompilée deux fois par pixel. Un échantillonneur
de pile les montre en une commande.

Mécanique : binaire SYMBOLISÉ bâti à part (`bench/prof-target`, même
optimisation que la release, non strippé — la release versionnée est strippée
et ne montre que des adresses), cas rendu à une taille assez grande pour durer,
piles échantillonnées par `sample` (macOS) ou `perf` (Linux), fonctions triées
par temps au SOMMET de pile, attente des workers exclue.

Sorties : `bench/harness/hotspots/<cas>.md` + tableau console.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .config import BENCH, REPO, TOML_DIR

PROF_TARGET = BENCH.parent / "prof-target"
PROF_CLI = PROF_TARGET / "release" / "fractall-cli"

# Sommets de pile qui ne sont pas du travail : workers rayon parqués ou en
# attente, démarrage du process. Exclus du dénominateur « temps occupé ».
IDLE_FRAMES = ("__psynch_cvwait", "swtch_pri", "cthread_yield",
               "semaphore_wait_trap", "mach_msg2_trap", "_dyld_start",
               "__workq_kernreturn", "start_wqthread", "__ulock_wait")

_RE_TOP = re.compile(r"^\s+(?P<sym>\S.*?)\s+\(in (?P<bin>[^)]+)\)"
                     r"(?:\s+load address.*?)?\s+(?P<n>\d+)\s*$")


def demangle(sym: str) -> str:
    """Démangle grossièrement un symbole Rust v0 (`_RNv…12fractall_cli7fractal…`)
    en chemin `fractal::bytecode::…` : suite d'identifiants préfixés par leur
    longueur. Les autres symboles (C, libsystem) sont rendus tels quels."""
    if not sym.startswith("_R"):
        return sym
    parts: list[str] = []
    i, n = 2, len(sym)
    while i < n:
        ch = sym[i]
        # Désambiguïsateurs (`s<base62>_`, dont le hash de crate `Cs…_`) et
        # rétro-références (`B<base62>_`) : de la syntaxe, pas des noms — et ils
        # contiennent des chiffres qu'il ne faut pas prendre pour une longueur.
        # Les identifiants étant consommés d'un bloc ci-dessous, un `s`/`B`
        # rencontré ICI est toujours de la syntaxe.
        if ch in "sB":
            j = i + 1
            while j < n and sym[j].isalnum():
                j += 1
            if j < n and sym[j] == "_":
                i = j + 1
                continue
        if ch.isdigit():
            j = i
            while j < n and sym[j].isdigit():
                j += 1
            length = int(sym[i:j])
            # `_` séparateur optionnel quand l'identifiant commence par un chiffre/_
            if j < n and sym[j] == "_":
                j += 1
            ident = sym[j:j + length]
            if 0 < length <= 80 and len(ident) == length and \
                    re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", ident):
                parts.append(ident)
                i = j + length
                continue
        i += 1
    # Le hash de crate (`Cs…_`) et les fermetures ne sont pas des identifiants.
    parts = [p for p in parts if p != "fractall_cli"]
    return "::".join(parts) if parts else sym


def parse_sample_top(text: str) -> list[tuple[str, str, int]]:
    """Section « Sort by top of stack » d'un rapport `sample` → [(symbole
    démanglé, binaire, échantillons)], ordre du rapport (décroissant)."""
    rows: list[tuple[str, str, int]] = []
    in_section = False
    for line in text.splitlines():
        if line.startswith("Sort by top of stack"):
            in_section = True
            continue
        if not in_section:
            continue
        if not line.strip():
            if rows:
                break
            continue
        m = _RE_TOP.match(line)
        if not m:
            continue
        sym = m["sym"].strip()
        if sym.startswith("???"):
            sym = "??? (non symbolisé)"
        rows.append((demangle(sym), m["bin"], int(m["n"])))
    return rows


def summarize(rows: list[tuple[str, str, int]], top: int = 15) -> dict:
    """Agrège par symbole, sépare l'attente du travail, calcule la part de
    chaque fonction dans le temps OCCUPÉ."""
    agg: dict[str, int] = {}
    idle = 0
    for sym, _bin, n in rows:
        if any(sym.startswith(f) for f in IDLE_FRAMES):
            idle += n
            continue
        agg[sym] = agg.get(sym, 0) + n
    busy = sum(agg.values())
    ranked = sorted(agg.items(), key=lambda kv: -kv[1])[:top]
    return {"busy": busy, "idle": idle,
            "rows": [(s, n, (100.0 * n / busy) if busy else 0.0) for s, n in ranked]}


def build_profiling_binary() -> bool:
    env = os.environ.copy()
    env.update({"CARGO_PROFILE_RELEASE_STRIP": "false",
                "CARGO_PROFILE_RELEASE_DEBUG": "line-tables-only"})
    cmd = ["cargo", "build", "--release", "--bin", "fractall-cli",
           "--target-dir", str(PROF_TARGET)]
    print(f"  $ {' '.join(cmd)}   (binaire symbolisé)", flush=True)
    return subprocess.run(cmd, cwd=REPO, env=env).returncode == 0


def _render_cmd(case: str, size: int, outdir: Path) -> list[str]:
    return [str(PROF_CLI), "--toml", str(TOML_DIR / f"{case}.toml"),
            "--width", str(size), "--height", str(size), "--bailout", "25",
            "--no-gpu", "--output", str(outdir / f"{case}.png")]


def sample_macos(case: str, size: int, seconds: float, delay: float,
                 outdir: Path) -> str | None:
    """Rend le cas et échantillonne ses piles avec `sample` (1 kHz)."""
    report = outdir / f"{case}.sample.txt"
    proc = subprocess.Popen(_render_cmd(case, size, outdir),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if delay > 0:
            time.sleep(delay)
        subprocess.run(["sample", str(proc.pid), str(max(1, round(seconds))), "1",
                        "-file", str(report)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        proc.wait()
    return report.read_text(errors="replace") if report.exists() else None


def cmd_hotspots(args) -> None:
    case = args.case
    if not (TOML_DIR / f"{case}.toml").exists():
        sys.exit(f"cas inconnu : toml/{case}.toml")
    outdir = BENCH / "hotspots"
    outdir.mkdir(parents=True, exist_ok=True)
    if not args.no_rebuild and not build_profiling_binary():
        sys.exit("échec du build du binaire symbolisé")
    if not PROF_CLI.exists():
        sys.exit(f"binaire symbolisé absent ({PROF_CLI}) — relancer sans --no-rebuild")

    if sys.platform != "darwin":
        # Linux : `perf` fait l'échantillonnage ; rapport rendu tel quel
        # (chemin non exercé sur la machine où cet outil a été écrit).
        if not shutil.which("perf"):
            sys.exit("`perf` introuvable — installer linux-perf, ou lancer "
                     "`perf record -g` à la main sur " + str(PROF_CLI))
        data = outdir / f"{case}.perf.data"
        subprocess.run(["perf", "record", "-F", "999", "-g", "-o", str(data), "--",
                        *_render_cmd(case, args.size, outdir)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rep = subprocess.run(["perf", "report", "--stdio", "--no-children",
                              "--sort", "symbol", "-i", str(data)],
                             capture_output=True, text=True).stdout
        (outdir / f"{case}.md").write_text("```\n" + rep[:8000] + "\n```\n")
        print("\n".join(rep.splitlines()[:60]))
        return

    text = sample_macos(case, args.size, args.seconds, args.delay, outdir)
    rows = parse_sample_top(text or "")
    summ = summarize(rows, top=args.top)
    if summ["busy"] < 200:
        print(f"⚠️  {summ['busy']} échantillons de travail seulement — le rendu "
              f"s'est terminé avant ou pendant l'attache de `sample` (~1 s). "
              f"Augmenter --size (actuel {args.size}) pour qu'il dure plusieurs "
              f"secondes.")
    L = [f"### hotspots `{case}` — {args.size}², {args.seconds:g} s échantillonnées",
         "",
         f"Échantillons : {summ['busy']} de travail, {summ['idle']} d'attente "
         f"(workers parqués, exclus des pourcentages).", "",
         "| % du temps occupé | échantillons | fonction (sommet de pile) |",
         "|---:|---:|---|"]
    for sym, n, pct in summ["rows"]:
        L.append(f"| {pct:.1f} | {n} | `{sym}` |")
    out = "\n".join(L) + "\n"
    (outdir / f"{case}.md").write_text(out)
    print(out)
    print(f"✓ {outdir / (case + '.md')}  (rapport brut : {case}.sample.txt)")
