"""Axe `truth` : le rendu de PRODUCTION contre le tier dd, sur le corpus.

Pourquoi : aucun autre axe ne compare la sortie par défaut à la vérité sur les
cas `toml/`. `quality` juge 16 presets figés ; `precision` juge l'orbite ;
`fuzz` juge des scènes shallow. Le 2026-11-06, cette comparaison faite à la
main a trouvé en une passe des centaines de pixels faux NON signalés dans le
rendu par défaut (e227 384 px à 256², golden_spider 169, leaded_glass 227 en
harmonic LA, 12 cas du tier exp) : le traqueur de fiabilité sous-estimait la
troncature des sauts BLA (ε au lieu de l·ε), sa borne exp retombait par
débordement d'exposant, et le chemin harmonic gardait un rayon de 2⁻²⁴.

Méthode, par cas : rendu de production → `.fmap` ; rendu juge `--dd-tier`
(frame entière en double-double ~106 b, elle-même gardée par son traqueur +
repli GMP par pixel, plafond de budget levé) ; comparaison des itérations
BRUTES. Le dd n'est pas la vérité absolue (cf. lethal_weapon : plancher du dd
à 1,6 M itérations) : un FAIL ici est une piste à adjuger contre le GMP pur
sur quelques pixels, pas un verdict définitif. Les paths sans orbite GMP
(`std_*`) sont `n/a`.
"""

from __future__ import annotations

import os
import subprocess
import time

from .axes_precision import (
    _PATH_RE, _PREC_RE, _render, read_fmap_iterations, verdict_for,
)
from .config import BENCH, CLI, TOML_DIR, cargo_build
from .guard import _classify_rc, journal_begin, journal_end

# Sentinelles : les scènes où le rendu par défaut était faux avant les
# correctifs du 2026-11-06 (f64 : e227, golden_spider, rug ; exp : olbaid2 ;
# harmonic LA : leaded_glass).
SENTINEL_CASES = ["e227", "golden_spider", "rug", "olbaid2", "leaded_glass"]


def last_path(text: str) -> str | None:
    """`path=` du DERNIER `[FRACTALL]` (une escalade dd en imprime deux)."""
    m = None
    for m in _PATH_RE.finditer(text):
        pass
    return m.group(1) if m else None


def is_dd_path(path: str | None) -> bool:
    return bool(path) and "dd" in path.split("_")


def count_diff(a: list[int], b: list[int]) -> int:
    return sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))


def axis_ddtruth(cases: list[str], size: int = 96, timeout: float = 300.0,
                 mem_limit_mb: int = 0, quarantined: set | None = None,
                 slow_safe: set | None = None) -> dict:
    if not CLI.exists():
        cargo_build("fractall-cli")
    if not CLI.exists():
        return {"status": "no_binary"}
    quarantined = quarantined or set()
    slow_safe = slow_safe or set()
    outdir = BENCH / "truth"
    outdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["FRACTALL_PERTURB_STATS"] = "1"
    # Juge : aucun pixel dd non fiable laissé sans recalcul GMP.
    judge_env = dict(env, FRACTALL_RELIABILITY_GMP_MAX_FRACTION="1")
    ordered = list(dict.fromkeys(list(cases) + SENTINEL_CASES))
    results: dict[str, dict] = {}
    for i, name in enumerate(ordered, 1):
        print(f"  [{i:>3}/{len(ordered)}] truth {name:22s}", end=" ", flush=True)
        if name in quarantined or name in slow_safe:
            results[name] = {"status": "skipped"}
            print("⊘ skip")
            continue
        toml_path = TOML_DIR / f"{name}.toml"
        if not toml_path.exists():
            results[name] = {"status": "missing_toml"}
            print("✗ toml absent")
            continue
        jrec = journal_begin("truth", name, {"size": f"{size}x{size}"})
        t0 = time.monotonic()
        prod_map = outdir / f"{name}.prod.fmap"
        try:
            proc = _render(toml_path, size, prod_map, [], env, timeout, mem_limit_mb)
        except subprocess.TimeoutExpired:
            journal_end(jrec, "timeout")
            results[name] = {"status": "timeout"}
            print("⏱ timeout")
            continue
        out = proc.stdout or ""
        path = last_path(out)
        if _classify_rc(proc.returncode) != "ok":
            journal_end(jrec, _classify_rc(proc.returncode))
            results[name] = {"status": "error"}
            print("✗ erreur")
            continue
        if _PREC_RE.search(out) is None or (path or "").startswith("std"):
            journal_end(jrec, "ok")
            results[name] = {"status": "n/a", "path": path}
            print(f"— hors perturbation ({path})")
            continue
        judge_map = outdir / f"{name}.dd.fmap"
        try:
            jproc = _render(toml_path, size, judge_map, ["--dd-tier"],
                            judge_env, timeout * 4, mem_limit_mb)
        except subprocess.TimeoutExpired:
            journal_end(jrec, "timeout")
            results[name] = {"status": "timeout", "path": path}
            print("⏱ timeout (juge dd)")
            continue
        status = _classify_rc(jproc.returncode)
        journal_end(jrec, status, {"secs": round(time.monotonic() - t0, 3)})
        if status != "ok":
            results[name] = {"status": status, "path": path}
            print(f"✗ juge {status}")
            continue
        # Le juge n'en est un que s'il a RÉELLEMENT rendu au tier dd. Pour un
        # type que le dd ne sait pas rendre (Julia, Burning Ship, hybrides,
        # opcodes — `wisdom::dd_tier_applicable`), `--dd-tier` refait le rendu
        # de production : 0 pixel d'écart, et un PASS qui ne juge rien.
        judge_path = last_path(jproc.stdout or "")
        if not is_dd_path(judge_path):
            results[name] = {"status": "n/a", "path": path,
                             "judge_path": judge_path}
            print(f"— pas de juge dd pour ce type ({path} ; juge {judge_path})")
            continue
        a = read_fmap_iterations(prod_map)
        b = read_fmap_iterations(judge_map)
        diff = count_diff(a, b)
        frac = diff / max(len(a), 1)
        verdict = verdict_for(frac)
        results[name] = {"status": "ok", "path": path, "diff_px": diff,
                         "frac": round(frac, 6), "verdict": verdict,
                         "sentinel": name in SENTINEL_CASES}
        print(f"{path} diff={diff} px ({100 * frac:.3f} %) {verdict}")
    ok = [r for r in results.values() if r.get("status") == "ok"]
    verdicts = [r["verdict"] for r in ok]
    return {
        "status": "ok", "size": size,
        "n_cases": len(ok), "n_pass": verdicts.count("PASS"),
        "n_warn": verdicts.count("WARN"), "n_fail": verdicts.count("FAIL"),
        "cases": results,
    }
