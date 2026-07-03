#!/usr/bin/env python3
"""Scorecard orchestrator — mesure fractall vs Fraktaler-3 (protocole HARNESS.md).

Produit `harness/history/<UTC>-<sha>.json` + `SCORECARD.md` (gaps triés) sur 4
axes : speed (ratio wall-clock vs F3), parity (compare_f3.py), quality
(fractall-quality suite vs GMP), goldens (cargo test golden_images).

  score [--tier quick|standard|full] [--axes ...] [--no-rebuild]
        [--cases a,b,c] [--width N] [--height N] [--runs N] [--timeout S]
  baseline   # fige le dernier history comme baseline
  gaps       # ré-affiche les gaps du dernier history

F3 auto-détecté (compare_f3.find_f3, override F3_BIN) ; sans lui speed/parity
sont `f3_unavailable` et le reste tourne.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_f3  # noqa: E402  (réutilise find_f3 / write_f3_wrapper / parse_light_toml)

REPO = Path(__file__).resolve().parent.parent
CLI = REPO / "target" / "release" / "fractall-cli"
QUALITY = REPO / "target" / "release" / "fractall-quality"
TOML_DIR = REPO / "toml"
HARNESS_DIR = REPO / "harness"
HISTORY_DIR = HARNESS_DIR / "history"
BASELINE = HARNESS_DIR / "baseline.json"
SCORECARD = REPO / "SCORECARD.md"
BENCH = REPO / "bench" / "harness"

SCHEMA = 1
SEVERITY_LABEL = {1: "correction", 2: "robustesse", 3: "vitesse", 4: "qualité"}
MARKER = "<!-- généré par scripts/harness.py — ne pas éditer à la main -->"

QUICK_CASES = [
    "test5", "spiral", "flake", "glitch_test_2", "e50", "e113", "e401",
    "e1000", "floral_fantasy", "dragon",
]
STANDARD_EXTRA = [
    "e318", "e1121", "e1200", "glitch_test_3", "glitch_test_4", "heaven",
    "integral_of_ex2", "windmill", "mitosis", "golden_spider", "leaded_glass",
    "magic", "tick_tock", "virus", "x",
]

def all_toml_stems() -> list[str]:
    return sorted(p.stem for p in TOML_DIR.glob("*.toml"))

def tier_config(tier: str) -> dict:
    # `quality_width/height` : résolution de l'axe quality (GMP pixel-par-pixel,
    # O(1e3-1e4) plus lent que la perturbation). Le tier `quick` la réduit à 96²
    # (~7× moins de pixels que 256²) pour que le cycle interne reste rapide — les
    # verdicts PASS/WARN/FAIL sont stables en résolution (bord chaotique inclus).
    if tier == "quick":
        return {"cases": list(QUICK_CASES), "width": 256, "height": 256,
                "runs": 1, "timeout": 120.0,
                "quality_width": 96, "quality_height": 96}
    if tier == "standard":
        return {"cases": QUICK_CASES + STANDARD_EXTRA, "width": 256,
                "height": 256, "runs": 3, "timeout": 300.0,
                "quality_width": 256, "quality_height": 256}
    if tier == "full":
        return {"cases": all_toml_stems(), "width": 256, "height": 256,
                "runs": 1, "timeout": 600.0,
                "quality_width": 256, "quality_height": 256}
    raise SystemExit(f"tier inconnu: {tier}")

def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"

def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, text=True
        )
        return bool(out.strip())
    except Exception:
        return False

def machine_info() -> dict:
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    return {"cpu": cpu, "nproc": os.cpu_count(),
            "os": f"{platform.system()} {platform.release()}"}

def cargo_build(bin_name: str | None = None) -> bool:
    cmd = ["cargo", "build", "--release"]
    if bin_name:
        cmd += ["--bin", bin_name]
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=REPO).returncode == 0

def timed_runs(cmd: list[str], env: dict, timeout: float, runs: int,
               accept_nonzero: bool = False) -> tuple[str, float | None]:
    """Chronomètre `cmd` `runs` fois (médiane). Renvoie (status, secondes|None).

    status : ok | timeout | fail. `accept_nonzero=True` enregistre quand même
    le temps si le process va au bout avec rc != 0 (cas F3 batch sans EXR :
    rendu complet mais save_exr no-op → rc != 0).
    """
    times: list[float] = []
    for _ in range(runs):
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env=env, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return "timeout", None
        except OSError:
            # binaire non exécutable ici (ex: build .macos lancé sur Linux)
            return "fail", None
        if proc.returncode != 0 and not accept_nonzero:
            return "fail", None
        times.append(time.monotonic() - t0)
    return "ok", statistics.median(times)

def axis_speed(cases: list[str], width: int, height: int, runs: int,
               timeout: float, f3_bin: Path | None) -> dict:
    outdir = BENCH / "speed"
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="harness_speed_"))
    results: dict[str, dict] = {}
    fr_env = os.environ.copy()
    fr_env["FRACTALL_NO_AUTO_ADJUST"] = "1"
    fr_env["FRACTALL_NO_PERIOD"] = "1"
    try:
        for i, name in enumerate(cases, 1):
            toml_path = TOML_DIR / f"{name}.toml"
            entry = {"fractall_s": None, "f3_s": None, "ratio": None,
                     "status": "ok", "f3_exr_ok": False}
            print(f"  [{i:>3}/{len(cases)}] speed {name:24s}", end=" ", flush=True)
            if not toml_path.exists():
                entry["status"] = "missing_toml"
                results[name] = entry
                print("✗ toml absent")
                continue

            fr_cmd = [str(CLI), "--toml", str(toml_path), "--width", str(width),
                      "--height", str(height), "--bailout", "25",
                      "--output", str(outdir / f"{name}.png")]
            st_fr, med_fr = timed_runs(fr_cmd, fr_env, timeout, runs)
            if st_fr == "ok":
                entry["fractall_s"] = round(med_fr, 4)
            elif st_fr == "timeout":
                entry["status"] = "fractall_timeout"
            else:
                entry["status"] = "fractall_fail"

            if f3_bin is None:
                # F3 absent : on garde le temps fractall, statut f3_unavailable
                if entry["status"] == "ok":
                    entry["status"] = "f3_unavailable"
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
            st_f3, med_f3 = timed_runs(f3_cmd, os.environ.copy(), timeout, runs,
                                       accept_nonzero=True)
            entry["f3_exr_ok"] = f3_exr.exists()
            if st_f3 == "ok":
                entry["f3_s"] = round(med_f3, 4)
            elif st_f3 == "timeout" and entry["status"] == "ok":
                entry["status"] = "f3_timeout"
            elif st_f3 == "fail" and entry["status"] == "ok":
                entry["status"] = "f3_fail"

            if entry["fractall_s"] and entry["f3_s"]:
                entry["ratio"] = round(entry["fractall_s"] / entry["f3_s"], 4)
            results[name] = entry
            print(f"fr={entry['fractall_s']}s f3={entry['f3_s']}s "
                  f"ratio={entry['ratio']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    ratios = [e["ratio"] for e in results.values() if e["ratio"] is not None]
    geomean = (math.exp(sum(math.log(r) for r in ratios) / len(ratios))
               if ratios else None)
    worst = None
    worst_pairs = [(n, e["ratio"]) for n, e in results.items()
                   if e["ratio"] is not None]
    if worst_pairs:
        n, r = max(worst_pairs, key=lambda x: x[1])
        worst = {"case": n, "ratio": round(r, 4)}
    wins = [n for n, e in results.items()
            if e["ratio"] is not None and e["ratio"] < 1.0]
    timeouts = [n for n, e in results.items()
                if e["status"] in ("fractall_timeout", "f3_timeout")]
    return {
        "status": "ok" if f3_bin else "f3_unavailable",
        "cases": results,
        "geomean_ratio": round(geomean, 4) if geomean else None,
        "worst_ratio": worst,
        "wins": wins,
        "timeouts": timeouts,
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
                f3_bin: Path | None) -> dict:
    empty = {"cases": {}, "n_ok": 0, "n_pixel_equiv": 0, "n_fail": 0,
             "n_timeout": 0, "n_f3_degenerate": 0}
    if f3_bin is None:
        return {"status": "f3_unavailable", **empty}
    outdir = BENCH / "parity"
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(REPO / "scripts" / "compare_f3.py"),
           "--only", ",".join(cases), "--width", str(width),
           "--height", str(height), "--timeout", str(timeout),
           "--out", str(outdir)]
    print(f"  $ compare_f3.py --only {len(cases)} cas", flush=True)
    subprocess.run(cmd, cwd=REPO)
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
    return {
        "status": axis_status,
        "cases": cases_out,
        "n_ok": n_ok,
        "n_pixel_equiv": n_pixel_equiv,
        "n_fail": n_fail,
        "n_timeout": n_timeout,
        "n_f3_degenerate": n_degen,
    }

def _qnum(it: dict, *keys):
    for k in keys:
        if k in it and it[k] is not None:
            try:
                return float(it[k])
            except (TypeError, ValueError):
                pass
    return None

def parse_quality_json(path: Path) -> list[dict] | None:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    items = None
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for k in ("presets", "rows", "results", "suite"):
            if isinstance(data.get(k), list):
                items = data[k]
                break
        if items is None and data and all(
                isinstance(v, dict) for v in data.values()):
            items = [{**v, "name": k} for k, v in data.items()]
    if items is None:
        return None
    rows = []
    for it in items:
        if not isinstance(it, dict):
            continue
        name = it.get("name") or it.get("preset") or "?"
        verdict = str(it.get("verdict") or it.get("status") or "").upper()
        rows.append({
            "name": name, "verdict": verdict,
            "max_iter_diff": _qnum(it, "max_iter_diff", "max"),
            "p99_iter_diff": _qnum(it, "p99_iter_diff", "p99"),
            "divergence_ratio": _qnum(it, "divergence_ratio",
                                      "iter_divergence_ratio"),
            "time_pert_ms": _qnum(it, "time_pert_ms", "perturb_time_ms"),
            "time_gmp_ms": _qnum(it, "time_gmp_ms", "gmp_time_ms"),
        })
    return rows or None

def parse_quality_md(text: str) -> list[dict] | None:
    """Fallback : colonnes de report.rs::write_suite_summary.

    | Preset | Verdict | max_iter_diff | p99_iter_diff | divergence_ratio |
      escape_disagree | max_|dz| | time_pert_ms | time_gmp_ms | speedup |
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 10:
            continue
        m = re.match(r"\[([^\]]+)\]", cells[0])
        if not m:  # entête / séparateur
            continue
        rows.append({
            "name": m.group(1),
            "verdict": cells[1].strip("* "),
            "max_iter_diff": _safe_float(cells[2]),
            "p99_iter_diff": _safe_float(cells[3]),
            "divergence_ratio": _safe_float(cells[4]),
            "time_pert_ms": _safe_float(cells[7]),
            "time_gmp_ms": _safe_float(cells[8]),
        })
    return rows or None

def _safe_float(s: str):
    try:
        return float(s.rstrip("x"))
    except (ValueError, AttributeError):
        return None

def summarize_quality(rows: list[dict]) -> dict:
    def has(r, tag):
        return r["verdict"].upper().startswith(tag)
    n_pass = sum(1 for r in rows if has(r, "PASS"))
    n_warn = sum(1 for r in rows if has(r, "WARN"))
    n_fail = sum(1 for r in rows if has(r, "FAIL"))
    return {
        "status": "ok",
        "n_pass": n_pass, "n_warn": n_warn, "n_fail": n_fail,
        "fail_presets": [r["name"] for r in rows if has(r, "FAIL")],
        "warn_presets": [r["name"] for r in rows if has(r, "WARN")],
        "presets": rows,
    }

def axis_quality(no_rebuild: bool, width: int = 256, height: int = 256) -> dict:
    if not QUALITY.exists() and not no_rebuild:
        cargo_build("fractall-quality")
    if not QUALITY.exists():
        # tentative de build même en --no-rebuild : sans binaire, rien à faire
        cargo_build("fractall-quality")
    if not QUALITY.exists():
        return {"status": "no_binary"}
    outdir = BENCH / "quality"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"  $ fractall-quality suite --width {width} --height {height}", flush=True)
    proc = subprocess.run([str(QUALITY), "suite", "--output-dir", str(outdir),
                           "--width", str(width), "--height", str(height)],
                          cwd=REPO)
    rows = None
    jf = outdir / "suite-summary.json"
    if jf.exists():
        rows = parse_quality_json(jf)
    if rows is None:
        md = outdir / "suite-summary.md"
        if md.exists():
            rows = parse_quality_md(md.read_text())
    if rows is None:
        return {"status": "parse_error", "returncode": proc.returncode}
    return summarize_quality(rows)

def axis_goldens() -> dict:
    print("  $ cargo test --release --test golden_images", flush=True)
    proc = subprocess.run(
        ["cargo", "test", "--release", "--test", "golden_images"],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    passed = proc.returncode == 0
    detail = "" if passed else "\n".join(proc.stdout.splitlines()[-40:])
    return {"passed": passed, "detail": detail}

def _gap(axis, case, metric, value, baseline_value, severity, note):
    return {"axis": axis, "case": case, "metric": metric, "value": value,
            "baseline_value": baseline_value, "severity": severity,
            "note": note}

def _comparable(card: dict, base: dict) -> bool:
    return (base.get("meta", {}).get("tier") == card["meta"]["tier"])

def compute_gaps(card: dict, base: dict | None) -> list[dict]:
    gaps: list[dict] = []
    # --- goldens (correction) ---
    g = card.get("goldens", {})
    if g and g.get("passed") is False:
        gaps.append(_gap("goldens", "", "passed", False, None, 1,
                         "golden images ROUGE"))
    # --- quality ---
    q = card.get("quality", {})
    for name in q.get("fail_presets", []):
        gaps.append(_gap("quality", name, "verdict", "FAIL", None, 1,
                         "quality suite FAIL"))
    for name in q.get("warn_presets", []):
        gaps.append(_gap("quality", name, "verdict", "WARN", None, 4,
                         "quality suite WARN"))
    # --- parity ---
    p = card.get("parity", {})
    for name, c in p.get("cases", {}).items():
        if c.get("status") == "fractall_fail":
            gaps.append(_gap("parity", name, "status", "fractall_fail", None,
                             1, "rendu fractall échoué"))
        rel = c.get("rel_dsi_pct")
        if rel is not None and rel > 2.0 and c.get("status") == "ok":
            gaps.append(_gap("parity", name, "rel_dsi_pct", round(rel, 4),
                             None, 2, "divergence parité >2%"))
    # --- speed (triées par ratio décroissant) ---
    s = card.get("speed", {})
    speed_gaps = []
    for name, c in s.get("cases", {}).items():
        r = c.get("ratio")
        if r is not None and r > 2.0:
            speed_gaps.append(_gap("speed", name, "ratio", round(r, 4), None,
                                   3, f"{r:.2f}× plus lent que F3"))
    speed_gaps.sort(key=lambda x: -x["value"])
    gaps += speed_gaps
    # --- régressions vs baseline (>10%) ---
    if base and _comparable(card, base):
        gaps += _regression_gaps(card, base)
    return gaps

def _regression_gaps(card: dict, base: dict) -> list[dict]:
    out = []
    cg = card.get("speed", {}).get("geomean_ratio")
    bg = base.get("speed", {}).get("geomean_ratio")
    if cg and bg and cg > bg * 1.10:
        out.append(_gap("speed", "<geomean>", "geomean_ratio", cg, bg, 3,
                        "RÉGRESSION vs baseline (>10%)"))
    cpe = card.get("parity", {}).get("n_pixel_equiv")
    bpe = base.get("parity", {}).get("n_pixel_equiv")
    if cpe is not None and bpe and cpe < bpe * 0.90:
        out.append(_gap("parity", "<agg>", "n_pixel_equiv", cpe, bpe, 2,
                        "RÉGRESSION vs baseline (>10%)"))
    cf = card.get("quality", {}).get("n_fail")
    bf = base.get("quality", {}).get("n_fail")
    if cf is not None and bf is not None and cf > bf:
        out.append(_gap("quality", "<agg>", "n_fail", cf, bf, 1,
                        "RÉGRESSION vs baseline"))
    return out

def _gap_mag(g: dict) -> float:
    try:
        return abs(float(g["value"]))
    except (TypeError, ValueError):
        return 0.0

def sort_gaps(gaps: list[dict]) -> list[dict]:
    return sorted(gaps, key=lambda g: (g["severity"], -_gap_mag(g)))

def _delta(cur, base, better="lower"):
    if cur is None or base is None:
        return ""
    d = cur - base
    if abs(d) < 1e-9:
        return " (=)"
    arrow = "↑" if d > 0 else "↓"
    good = (d < 0) if better == "lower" else (d > 0)
    tag = "✅" if good else "⚠️"
    return f" ({arrow}{abs(d):.3g} {tag})"

def build_scorecard_md(card: dict, base: dict | None) -> str:
    m = card["meta"]
    comparable = bool(base and _comparable(card, base))
    b = base if comparable else None
    L = [MARKER, "", "# SCORECARD — fractall vs Fraktaler-3", ""]
    L.append(f"- **Date** : {m['date_utc']}")
    L.append(f"- **Commit** : `{m['git_sha']}`"
             + ("  ⚠️ arbre modifié (dirty)" if m["git_dirty"] else ""))
    mach = m["machine"]
    L.append(f"- **Machine** : {mach['cpu']} · {mach['nproc']} threads · "
             f"{mach['os']}")
    qw, qh = m.get("quality_width", m["width"]), m.get("quality_height", m["height"])
    qual_note = "" if (qw, qh) == (m["width"], m["height"]) else f" · quality {qw}×{qh}"
    L.append(f"- **Tier** : {m['tier']} · {m['width']}×{m['height']}{qual_note} · "
             f"runs={m['runs']} · axes={','.join(m['axes'])}")
    L.append(f"- **F3** : {m['f3_bin'] or '— (indisponible)'}")
    if base and not comparable:
        L.append(f"- _baseline présente mais tier différent "
                 f"({base.get('meta', {}).get('tier')}) — pas de delta._")
    L.append("")

    # Speed
    s = card.get("speed")
    if s and s.get("status") != "skipped":
        L += ["## Vitesse (ratio fractall/F3, <1 = fractall gagne)", ""]
        if s.get("status") == "f3_unavailable":
            L.append("_F3 indisponible — timings fractall seuls "
                     "(voir history JSON)._\n")
        bs = (b or {}).get("speed", {})
        L.append("| Métrique | Valeur | vs baseline |")
        L.append("|---|---:|---|")
        L.append(f"| geomean ratio | {_fmt(s.get('geomean_ratio'))} | "
                 f"{_delta(s.get('geomean_ratio'), bs.get('geomean_ratio'))} |")
        wr = s.get("worst_ratio")
        worst_txt = f"{_fmt(wr['ratio'])} ({wr['case']})" if wr else "—"
        L.append(f"| pire ratio | {worst_txt} | |")
        L.append(f"| wins (ratio<1) | {len(s.get('wins', []))} | |")
        L.append(f"| timeouts | {len(s.get('timeouts', []))} | |")
        L.append(f"| cas comparés | {s.get('n_compared', 0)}/"
                 f"{s.get('n_cases', 0)} | |")
        L.append("")

    # Parity
    p = card.get("parity")
    if p and p.get("status") != "skipped":
        L += ["## Parité (compare_f3 — Δsmooth-iter vs F3)", ""]
        if p.get("status") == "f3_unavailable":
            L.append("_F3 indisponible — axe non exécuté._\n")
        elif p.get("status") == "f3_no_exr":
            L.append("_F3 build Linux sans support EXR (HAVE_EXR=0) — parité "
                     "impossible tant que F3 n'est pas rebuilé avec OpenEXR._\n")
        else:
            bp = (b or {}).get("parity", {})
            L.append("| Métrique | Valeur | vs baseline |")
            L.append("|---|---:|---|")
            for key, label, better in [
                ("n_ok", "n_ok", "higher"),
                ("n_pixel_equiv", "pixel-équivalents (<0.01%)", "higher"),
                ("n_fail", "échecs", "lower"),
                ("n_timeout", "timeouts", "lower"),
                ("n_f3_degenerate", "F3-dégénéré (win fractall)", "higher"),
            ]:
                L.append(f"| {label} | {p.get(key, 0)} | "
                         f"{_delta(p.get(key), bp.get(key), better)} |")
            L.append("")

    # Quality
    q = card.get("quality")
    if q and q.get("status") != "skipped":
        L += ["## Qualité (fractall-quality suite — perturbation vs GMP)", ""]
        if q.get("status") in ("no_binary", "parse_error"):
            L.append(f"_axe indisponible : {q.get('status')}._\n")
        else:
            bq = (b or {}).get("quality", {})
            L.append("| Verdict | Nombre | vs baseline |")
            L.append("|---|---:|---|")
            L.append(f"| PASS | {q.get('n_pass', 0)} | "
                     f"{_delta(q.get('n_pass'), bq.get('n_pass'), 'higher')} |")
            L.append(f"| WARN | {q.get('n_warn', 0)} | "
                     f"{_delta(q.get('n_warn'), bq.get('n_warn'), 'lower')} |")
            L.append(f"| FAIL | {q.get('n_fail', 0)} | "
                     f"{_delta(q.get('n_fail'), bq.get('n_fail'), 'lower')} |")
            if q.get("fail_presets"):
                L.append(f"\nFAIL : {', '.join(q['fail_presets'])}")
            L.append("")

    # Goldens
    g = card.get("goldens")
    if g and g.get("status") != "skipped":
        status = "🟢 VERT" if g.get("passed") else "🔴 ROUGE"
        L += ["## Goldens (pixel-exact)", "", f"- {status}", ""]

    # beyond_f3
    bf = card.get("beyond_f3", {})
    if bf.get("speed_wins") or bf.get("correctness_wins"):
        L += ["## Au-delà de F3", ""]
        if bf.get("speed_wins"):
            L.append(f"- **speed_wins** : {', '.join(bf['speed_wins'])}")
        if bf.get("correctness_wins"):
            L.append(f"- **correctness_wins** (F3 dégénéré rendu par "
                     f"fractall) : {', '.join(bf['correctness_wins'])}")
        L.append("")

    # Gaps top-10
    gaps = sort_gaps(card.get("gaps", []))
    L += ["## Gaps (top 10 — sévérité asc, magnitude desc)", ""]
    if not gaps:
        L.append("_aucun gap détecté 🎉_")
    else:
        L.append("| # | Sévérité | Axe | Cas | Métrique | Valeur | Note |")
        L.append("|---:|---|---|---|---|---:|---|")
        for i, gp in enumerate(gaps[:10], 1):
            sev = f"{gp['severity']} {SEVERITY_LABEL[gp['severity']]}"
            L.append(f"| {i} | {sev} | {gp['axis']} | `{gp['case']}` | "
                     f"{gp['metric']} | {gp['value']} | {gp['note']} |")
        if len(gaps) > 10:
            L.append(f"\n_… {len(gaps) - 10} gap(s) supplémentaire(s) dans "
                     f"le history JSON._")
    L.append("")
    L.append("---")
    L.append(f"_Scorecards versionnés : `harness/history/` · baseline : "
             f"`harness/baseline.json`. Généré par `scripts/harness.py`._")
    return "\n".join(L) + "\n"

def _fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"

def latest_history() -> Path | None:
    files = sorted(HISTORY_DIR.glob("*.json"))
    return files[-1] if files else None

def write_history(card: dict) -> Path:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    m = card["meta"]
    path = HISTORY_DIR / f"{m['stamp']}-{m['git_sha']}.json"
    path.write_text(json.dumps(card, indent=2, ensure_ascii=False))
    return path

def cmd_score(args) -> None:
    cfg = tier_config(args.tier)
    cases = ([c.strip() for c in args.cases.split(",")] if args.cases
             else cfg["cases"])
    width = args.width or cfg["width"]
    height = args.height or cfg["height"]
    runs = args.runs or cfg["runs"]
    timeout = args.timeout or cfg["timeout"]
    axes = [a.strip() for a in args.axes.split(",") if a.strip()]

    if not args.no_rebuild:
        print("→ cargo build --release", flush=True)
        if not cargo_build():
            sys.exit("cargo build a échoué")

    f3_bin = compare_f3.find_f3()
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")

    card: dict = {
        "schema": SCHEMA,
        "meta": {
            "date_utc": now.replace(microsecond=0).isoformat(),
            "stamp": stamp,
            "git_sha": git_sha(),
            "git_dirty": git_dirty(),
            "machine": machine_info(),
            "tier": args.tier,
            "axes": axes,
            "width": width,
            "height": height,
            "quality_width": cfg.get("quality_width", 256),
            "quality_height": cfg.get("quality_height", 256),
            "runs": runs,
            "f3_bin": str(f3_bin) if f3_bin else None,
        },
        "speed": {"status": "skipped"},
        "parity": {"status": "skipped"},
        "quality": {"status": "skipped"},
        "goldens": {"status": "skipped"},
    }

    print(f"→ tier={args.tier} cases={len(cases)} {width}×{height} "
          f"runs={runs} timeout={timeout}s axes={axes}")
    print(f"→ F3: {f3_bin or 'INDISPONIBLE (speed/parity dégradés)'}")

    if "speed" in axes:
        print("\n== SPEED ==")
        card["speed"] = axis_speed(cases, width, height, runs, timeout, f3_bin)
    if "parity" in axes:
        print("\n== PARITY ==")
        card["parity"] = axis_parity(cases, width, height, timeout, f3_bin)
    if "quality" in axes:
        print("\n== QUALITY ==")
        card["quality"] = axis_quality(
            args.no_rebuild,
            cfg.get("quality_width", 256),
            cfg.get("quality_height", 256),
        )
    if "goldens" in axes:
        print("\n== GOLDENS ==")
        card["goldens"] = axis_goldens()

    card["beyond_f3"] = {
        "speed_wins": card.get("speed", {}).get("wins", []),
        "correctness_wins": [
            n for n, c in card.get("parity", {}).get("cases", {}).items()
            if c.get("status") == "f3_degenerate"],
    }

    base = None
    if BASELINE.exists():
        try:
            base = json.loads(BASELINE.read_text())
        except Exception:
            base = None
    card["gaps"] = compute_gaps(card, base)

    hist = write_history(card)
    SCORECARD.write_text(build_scorecard_md(card, base))
    print(f"\n✓ history : {hist.relative_to(REPO)}")
    print(f"✓ scorecard : {SCORECARD.relative_to(REPO)}")
    _print_gaps(card["gaps"])

def cmd_baseline(_args) -> None:
    latest = latest_history()
    if latest is None:
        sys.exit("Aucun history — lance d'abord `harness.py score`.")
    shutil.copyfile(latest, BASELINE)
    print(f"✓ baseline ← {latest.relative_to(REPO)}")

def cmd_gaps(_args) -> None:
    latest = latest_history()
    if latest is None:
        sys.exit("Aucun history — lance d'abord `harness.py score`.")
    card = json.loads(latest.read_text())
    print(f"Gaps du dernier score ({latest.name}) :\n")
    _print_gaps(card.get("gaps", []))

def _print_gaps(gaps: list[dict]) -> None:
    gaps = sort_gaps(gaps)
    if not gaps:
        print("Aucun gap détecté 🎉")
        return
    print(f"\n{len(gaps)} gap(s) — top 10 :")
    for i, g in enumerate(gaps[:10], 1):
        sev = SEVERITY_LABEL[g["severity"]]
        print(f"  {i:>2}. [{g['severity']} {sev:<10}] {g['axis']}/{g['case']} "
              f"{g['metric']}={g['value']} — {g['note']}")

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    sc = sub.add_parser("score", help="Mesure et écrit un scorecard.")
    sc.add_argument("--tier", choices=["quick", "standard", "full"],
                    default="quick")
    sc.add_argument("--axes", default="speed,parity,quality,goldens")
    sc.add_argument("--no-rebuild", action="store_true")
    sc.add_argument("--cases", default=None,
                    help="liste CSV de stems (override du tier)")
    sc.add_argument("--width", type=int, default=None)
    sc.add_argument("--height", type=int, default=None)
    sc.add_argument("--runs", type=int, default=None)
    sc.add_argument("--timeout", type=float, default=None)
    sc.set_defaults(func=cmd_score)

    bl = sub.add_parser("baseline", help="Fige le dernier history comme baseline.")
    bl.set_defaults(func=cmd_baseline)

    gp = sub.add_parser("gaps", help="Ré-affiche les gaps du dernier history.")
    gp.set_defaults(func=cmd_gaps)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
