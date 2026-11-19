"""Volet « path f64 standard » de l'axe `truth` (chantier de correction n° 1).

Pourquoi : le mode Auto rend les vues peu profondes (pixel > ~1e-12) par le
path f64 STANDARD, qui n'avait aucun suivi d'erreur ; sur un bord chaotique à
fort plafond il rendait des comptes faux sans rien signaler (seahorse 1e10,
2500 it., 256×160 : 7,8 % de pixels faux contre GMP pur, mesuré le
2026-11-06). Le volet dd de `truth` marque ces cas `n/a` (pas d'orbite GMP),
`quality` et `fuzz` forcent la perturbation : rien ne les voyait.

Méthode, par scène (lieu × zoom × plafond) : rendu de PRODUCTION (Auto,
`--no-gpu`, 256×160 — la basse résolution est la plus exposée : la bascule
Auto → perturbation dépend de la taille du pixel) → `.fmap` ; juge
`--algorithm gmp` (GMP pur par pixel, 256 b à ces profondeurs : convergé,
vérifié ×2 == au pixel près sur seahorse 1e10) ; comparaison des itérations
brutes, mêmes seuils que les autres volets (PASS ≤ 0,05 %, FAIL ≥ 0,5 %).
"""

from __future__ import annotations

import os
import subprocess
import time

from .axes_precision import _PATH_RE, read_fmap_iterations, verdict_for
from .config import BENCH, CLI
from .guard import _classify_rc, _rlimit_preexec, journal_begin, journal_end

# Lieux : spirale (seahorse, le pire), vallée des éléphants (point de la
# bande d'évasion à 1500 it. obtenu par bissection sur y = 0,0085 — un point
# du bord « à l'infini » serait de l'intérieur à 2500 it.), minibrot de l'axe réel
# (période 3 : surtout de l'intérieur, témoin à peu de pixels faux).
LOCATIONS = {
    "seahorse": ("-0.743643887037158704752191506114774",
                 "0.131825904205311970493132056385139"),
    "elephants": ("0.27631692275258796", "0.0085"),
    "minibrot": ("-1.7548776662466927", "0.0"),
}
ZOOMS = ["1e4", "1e7", "1e10"]
ITERATIONS = [1000, 2500]
WIDTH, HEIGHT = 256, 160


def scene_name(loc: str, zoom: str, iters: int) -> str:
    return f"std:{loc}_{zoom}_{iters}"


# Scènes au plafond que CHOISIT la sonde d'itérations automatiques (le
# régime réel d'un utilisateur « Iter: Auto », pas le défaut 2500). Trouvée par
# l'axe `wisdom` le 2026-11-19 : éléphants 1e7 à 60 000 itérations, 43 % de
# pixels signalés, 1,04 % de pixels FAUX vs GMP pur (juge convergé : 256 b ==
# 640 b à 3 px près) — invisible à 2500. Jugées dans tous les tiers.
AUTO_CAP_SCENES = [("elephants", "1e7", 60000)]


def scenes(full: bool) -> list[tuple[str, str, int]]:
    """Scènes jugées. Tier quick : le plafond par défaut (2500) seulement,
    plus les scènes à plafond automatique."""
    iters = ITERATIONS if full else [2500]
    return ([(loc, z, it) for loc in LOCATIONS for z in ZOOMS for it in iters]
            + AUTO_CAP_SCENES)


def _cmd(loc: str, zoom: str, iters: int, out_map, extra: list[str]) -> list[str]:
    cx, cy = LOCATIONS[loc]
    return [str(CLI), "--type", "3", f"--center-x-hp={cx}", f"--center-y-hp={cy}",
            "--zoom", zoom, "--iterations", str(iters),
            "--width", str(WIDTH), "--height", str(HEIGHT), "--no-gpu",
            "--output", str(out_map.with_suffix(".png")),
            "--output-map", str(out_map)] + extra


def _last_path(text: str) -> str | None:
    m = None
    for m in _PATH_RE.finditer(text):
        pass
    return m.group(1) if m else None


def axis_stdtruth(full: bool, timeout: float = 300.0,
                  mem_limit_mb: int = 0) -> dict[str, dict]:
    """Résultats par scène, au format des cas de l'axe `truth`."""
    outdir = BENCH / "truth-std"
    outdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["FRACTALL_PERTURB_STATS"] = "1"
    results: dict[str, dict] = {}
    todo = scenes(full)
    for i, (loc, zoom, iters) in enumerate(todo, 1):
        name = scene_name(loc, zoom, iters)
        print(f"  [{i:>3}/{len(todo)}] truth {name:28s}", end=" ", flush=True)
        jrec = journal_begin("truth", name, {"size": f"{WIDTH}x{HEIGHT}"})
        stem = name.replace(":", "_")
        prod_map = outdir / f"{stem}.prod.fmap"
        judge_map = outdir / f"{stem}.gmp.fmap"
        try:
            t0 = time.monotonic()
            proc = subprocess.run(
                _cmd(loc, zoom, iters, prod_map, []), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, env=env, timeout=timeout, text=True,
                errors="replace", preexec_fn=_rlimit_preexec(mem_limit_mb))
            prod_secs = time.monotonic() - t0
            jproc = subprocess.run(
                _cmd(loc, zoom, iters, judge_map, ["--algorithm", "gmp"]),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                timeout=timeout * 4, text=True, errors="replace",
                preexec_fn=_rlimit_preexec(mem_limit_mb))
        except subprocess.TimeoutExpired:
            journal_end(jrec, "timeout")
            results[name] = {"status": "timeout"}
            print("⏱ timeout")
            continue
        status = _classify_rc(proc.returncode)
        if status == "ok":
            status = _classify_rc(jproc.returncode)
        journal_end(jrec, status)
        if status != "ok":
            results[name] = {"status": status}
            print(f"✗ {status}")
            continue
        path = _last_path(proc.stdout or "")
        a = read_fmap_iterations(prod_map)
        b = read_fmap_iterations(judge_map)
        diff = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
        frac = diff / max(len(a), 1)
        verdict = verdict_for(frac)
        results[name] = {"status": "ok", "path": path, "diff_px": diff,
                         "frac": round(frac, 6), "verdict": verdict,
                         "secs": round(prod_secs, 3), "judge": "gmp",
                         "sentinel": False}
        print(f"{path} diff={diff} px ({100 * frac:.3f} %) {verdict} "
              f"[{prod_secs:.2f} s]")
    return results
