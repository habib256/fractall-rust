"""Axes mesurés CONTRE Fraktaler-3 : vitesse (ratio wall-clock) et parité (compare_f3)."""

from __future__ import annotations

from pathlib import Path
import csv
import math
import os
import shutil
import subprocess
import sys
import tempfile

from .config import BENCH, CLI, REPO, TOML_DIR, load_compare_f3
from .guard import (
    timed_runs_detail,
    _classify_rc, _rlimit_preexec, add_quarantine, journal_begin, journal_end,
    load_f3_degenerate, timed_runs, update_f3_degenerate,
)

def axis_speed(cases: list[str], width: int, height: int, runs: int,
               timeout: float, f3_bin: Path | None,
               mem_limit_mb: int = 0, quarantined: set | None = None,
               slow_safe: set | None = None) -> dict:
    outdir = BENCH / "speed"
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="harness_speed_"))
    results: dict[str, dict] = {}
    quarantined = quarantined or set()
    slow_safe = slow_safe or set()
    fr_env = os.environ.copy()
    fr_env["FRACTALL_NO_AUTO_ADJUST"] = "1"
    fr_env["FRACTALL_NO_PERIOD"] = "1"
    # F3 présent mais deps Python du wrapper (compare_f3 : numpy/OpenEXR)
    # absentes → on mesure fractall seul, statut distinct `f3_deps_missing`.
    compare_f3 = load_compare_f3() if f3_bin else None
    f3_deps_missing = f3_bin is not None and compare_f3 is None
    if f3_deps_missing:
        f3_bin = None
    try:
        for i, name in enumerate(cases, 1):
            toml_path = TOML_DIR / f"{name}.toml"
            entry = {"fractall_s": None, "f3_s": None, "ratio": None,
                     "status": "ok", "f3_exr_ok": False}
            print(f"  [{i:>3}/{len(cases)}] speed {name:24s}", end=" ", flush=True)
            if name in quarantined:
                entry["status"] = "quarantined"
                results[name] = entry
                print("⊘ QUARANTAINE (skip — voir quarantine.json)")
                continue
            if name in slow_safe:
                # correct + memory-safe mais hors budget temps → skip SANS gap
                # robustesse (ratio None, exclu des agrégats comme f3_degenerate).
                entry["status"] = "slow_safe"
                results[name] = entry
                print("⊘ SLOW-SAFE (skip — correct mais hors budget, "
                      "voir slow-safe.json)")
                continue
            if not toml_path.exists():
                entry["status"] = "missing_toml"
                results[name] = entry
                print("✗ toml absent")
                continue

            fr_cmd = [str(CLI), "--toml", str(toml_path), "--width", str(width),
                      "--height", str(height), "--bailout", "25",
                      "--output", str(outdir / f"{name}.png")]
            jrec = journal_begin("speed", name,
                                 {"size": f"{width}x{height}",
                                  "toml": str(toml_path)})
            st_fr, det_fr = timed_runs_detail(fr_cmd, fr_env, timeout, runs,
                                              mem_limit_mb=mem_limit_mb)
            med_fr = det_fr["median"] if det_fr else None
            journal_end(jrec, st_fr, {"secs": med_fr})
            if st_fr == "ok":
                entry["fractall_s"] = round(med_fr, 4)
                entry["fractall_s_min"] = round(det_fr["min"], 4)
                entry["fractall_s_spread"] = det_fr["spread"]
            elif st_fr == "timeout":
                entry["status"] = "fractall_timeout"
            elif st_fr in ("killed_oom", "aborted", "killed"):
                entry["status"] = f"fractall_{st_fr}"
                # gourmand mémoire tué par le cap → quarantaine auto pour ne pas
                # replanter le sweep suivant ; l'incident reste dans le journal.
                if add_quarantine(name, f"{st_fr} sous cap mémoire "
                                        f"{mem_limit_mb}MB @ {width}x{height}"):
                    print("⚠ tué (mémoire) → QUARANTAINE  ", end="")
            else:
                entry["status"] = "fractall_fail"

            if f3_bin is None:
                # F3 absent : on garde le temps fractall, statut f3_unavailable
                if entry["status"] == "ok":
                    entry["status"] = ("f3_deps_missing" if f3_deps_missing
                                       else "f3_unavailable")
                results[name] = entry
                fs = entry["fractall_s"]
                print(f"⊘ fr={fs}s (F3 indispo)")
                continue

            src = compare_f3.parse_light_toml(toml_path)
            iters = src.iterations or 1024
            if iters > 2 ** 31 - 1:
                iters = 2 ** 31 - 1
            wrapper = compare_f3.write_f3_wrapper(
                src, tmp, name, width, height, iters, 25.0)
            f3_exr = tmp / f"{name}_f3.exr"
            if f3_exr.exists():
                f3_exr.unlink()
            f3_cmd = [str(f3_bin), "-b", "-P", str(wrapper)]
            # accept_nonzero : le build Linux (HAVE_EXR=0) rend complètement mais
            # exit rc != 0 (save_exr no-op) — on garde le timing si pas timeout.
            jrec_f3 = journal_begin("speed_f3", name,
                                    {"size": f"{width}x{height}"})
            st_f3, med_f3 = timed_runs(f3_cmd, os.environ.copy(), timeout, runs,
                                       accept_nonzero=True,
                                       mem_limit_mb=mem_limit_mb)
            journal_end(jrec_f3, st_f3, {"secs": med_f3})
            entry["f3_exr_ok"] = f3_exr.exists()
            if st_f3 == "ok":
                entry["f3_s"] = round(med_f3, 4)
            elif st_f3 == "timeout" and entry["status"] == "ok":
                entry["status"] = "f3_timeout"
            elif st_f3 in ("fail", "killed_oom", "aborted", "killed") \
                    and entry["status"] == "ok":
                entry["status"] = "f3_fail" if st_f3 == "fail" else f"f3_{st_f3}"

            if entry["fractall_s"] and entry["f3_s"]:
                entry["ratio"] = round(entry["fractall_s"] / entry["f3_s"], 4)
            results[name] = entry
            print(f"fr={entry['fractall_s']}s f3={entry['f3_s']}s "
                  f"ratio={entry['ratio']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # Exclure les cas F3-dégénérés des AGRÉGATS (geomean/pire/wins) : F3 y rend
    # un fast-path uniforme FAUX (cf. registre f3-degenerate.json, entretenu par
    # l'axe parité) → le ratio mesuré est un artefact, pas un gap moteur
    # (glitch_test_5 3.8× fantôme, 2026-07-12). Le ratio mesuré reste stocké
    # dans l'entrée (flag `f3_degenerate`) et listé à part dans le scorecard.
    degen = set(load_f3_degenerate())
    for n, e in results.items():
        if n in degen:
            e["f3_degenerate"] = True
    excluded_degen = [n for n in results if n in degen
                      and results[n]["ratio"] is not None]
    ratios = [e["ratio"] for n, e in results.items()
              if e["ratio"] is not None and n not in degen]
    geomean = (math.exp(sum(math.log(r) for r in ratios) / len(ratios))
               if ratios else None)
    worst = None
    worst_pairs = [(n, e["ratio"]) for n, e in results.items()
                   if e["ratio"] is not None and n not in degen]
    if worst_pairs:
        n, r = max(worst_pairs, key=lambda x: x[1])
        worst = {"case": n, "ratio": round(r, 4)}
    wins = [n for n, e in results.items()
            if e["ratio"] is not None and e["ratio"] < 1.0 and n not in degen]
    timeouts = [n for n, e in results.items()
                if e["status"] in ("fractall_timeout", "f3_timeout")]
    measured = [e["fractall_s"] for e in results.values()
                if e.get("fractall_s") is not None]
    return {
        # Vitesse INTRINSÈQUE (sans F3) : temps fractall cumulé + nb mesurés —
        # comparés par cas à la baseline même machine (gaps.self_speed_ratio).
        "fractall_total_s": round(sum(measured), 4),
        "n_measured": len(measured),
        "status": ("ok" if f3_bin else
                   "f3_deps_missing" if f3_deps_missing else "f3_unavailable"),
        "cases": results,
        "geomean_ratio": round(geomean, 4) if geomean else None,
        "worst_ratio": worst,
        "wins": wins,
        "timeouts": timeouts,
        "excluded_f3_degenerate": excluded_degen,
        "excluded_slow_safe": sorted(
            n for n, e in results.items() if e["status"] == "slow_safe"),
        "n_cases": len(cases),
        "n_compared": len(ratios),
    }

def _fnum(row: dict, key: str):
    v = row.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None

def axis_parity(cases: list[str], width: int, height: int, timeout: float,
                f3_bin: Path | None, mem_limit_mb: int = 0) -> dict:
    empty = {"cases": {}, "n_ok": 0, "n_pixel_equiv": 0, "n_fail": 0,
             "n_timeout": 0, "n_f3_degenerate": 0}
    if f3_bin is None:
        return {"status": "f3_unavailable", **empty}
    if load_compare_f3() is None:
        return {"status": "f3_deps_missing", **empty}
    outdir = BENCH / "parity"
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(REPO / "scripts" / "compare_f3.py"),
           "--only", ",".join(cases), "--width", str(width),
           "--height", str(height), "--timeout", str(timeout),
           "--out", str(outdir)]
    print(f"  $ compare_f3.py --only {len(cases)} cas", flush=True)
    jrec = journal_begin("parity", "<compare_f3-suite>",
                         {"size": f"{width}x{height}", "n_cases": len(cases)})
    proc = subprocess.run(cmd, cwd=REPO, preexec_fn=_rlimit_preexec(mem_limit_mb))
    journal_end(jrec, _classify_rc(proc.returncode))
    csv_path = outdir / "_summary.csv"
    if not csv_path.exists():
        return {"status": "error", **empty}

    cases_out: dict[str, dict] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            name = row.get("name", "?")
            cases_out[name] = {
                "status": row.get("status", "?"),
                "rel_dsi_pct": _fnum(row, "rel_dsi_pct"),
                "mean_abs_dsi": _fnum(row, "mean_abs_dsi"),
                "inside_mismatch": int(row["inside_mismatch"])
                if row.get("inside_mismatch") else None,
                "f3_degenerate": row.get("status") == "f3_degenerate",
                "f3_secs_num": _fnum(row, "f3_secs_num"),
                "fr_secs_num": _fnum(row, "fr_secs_num"),
            }

    def st(name):
        return cases_out[name]["status"]

    n_ok = sum(1 for n in cases_out if st(n) == "ok")
    n_pixel_equiv = sum(
        1 for n, c in cases_out.items()
        if c["status"] == "ok" and c["rel_dsi_pct"] is not None
        and c["rel_dsi_pct"] < 0.01)
    n_timeout = sum(1 for n in cases_out
                    if st(n) in ("fractall_timeout", "f3_timeout"))
    n_degen = sum(1 for n in cases_out if st(n) == "f3_degenerate")
    n_fail = sum(1 for n in cases_out if st(n) not in (
        "ok", "f3_degenerate", "fractall_timeout", "f3_timeout"))
    # Heuristique : F3 sans support EXR (build Linux HAVE_EXR=0) → tous les cas
    # tombent en f3_fail (aucun .exr produit). On le signale distinctement pour
    # ne pas confondre avec un vrai bug de rendu (rebuild EXR en cours).
    statuses = [c["status"] for c in cases_out.values()]
    axis_status = "ok"
    if (statuses and n_ok == 0 and n_degen == 0
            and all(x in ("f3_fail", "f3_timeout") for x in statuses)
            and any(x == "f3_fail" for x in statuses)):
        axis_status = "f3_no_exr"
    # Persister les cas F3-dégénérés pour l'axe vitesse (qui peut tourner seul) :
    # leurs ratios n'ont pas de sens (F3 fast-path faux vs rendu correct).
    if axis_status == "ok":
        update_f3_degenerate(cases_out)
    return {
        "status": axis_status,
        "cases": cases_out,
        "n_ok": n_ok,
        "n_pixel_equiv": n_pixel_equiv,
        "n_fail": n_fail,
        "n_timeout": n_timeout,
        "n_f3_degenerate": n_degen,
    }
