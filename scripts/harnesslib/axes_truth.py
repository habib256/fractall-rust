"""Axes mesurés contre la VÉRITÉ GMP : quality suite, fuzz, goldens pixel-exact."""

from __future__ import annotations

from pathlib import Path
import json
import re
import subprocess
import sys

from .config import BENCH, QUALITY, REPO, cargo_build
from .guard import _classify_rc, _rlimit_preexec, journal_begin, journal_end

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

def axis_quality(no_rebuild: bool, width: int = 256, height: int = 256,
                 mem_limit_mb: int = 0) -> dict:
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
    jrec = journal_begin("quality", "<quality-suite>",
                         {"size": f"{width}x{height}"})
    proc = subprocess.run([str(QUALITY), "suite", "--output-dir", str(outdir),
                           "--width", str(width), "--height", str(height)],
                          cwd=REPO, preexec_fn=_rlimit_preexec(mem_limit_mb))
    journal_end(jrec, _classify_rc(proc.returncode))
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

def axis_fuzz(probes: int, seed: int, width: int = 96, height: int = 96,
              mem_limit_mb: int = 0) -> dict:
    """Sondes aléatoires DÉTERMINISTES pert-vs-GMP (G8.2). Les scènes sont des
    points frontière stables (échappement invariant à un bump 2⁻⁴⁶ de c —
    ground truth bien conditionnée, cf. scripts/fuzz_scenes.py) tirés sur les
    6 familles Mandelbrot-like, zoom 1e5-1e11. Vise les classes de divergence
    que les presets figés ratent (ex. e13/dd-sensibilité, trouvée par accident).
    Seed committée dans le scorecard → reproductible."""
    if not QUALITY.exists():
        cargo_build("fractall-quality")
    if not QUALITY.exists():
        return {"status": "no_binary"}
    sys.path.insert(0, str(REPO / "scripts"))
    import fuzz_scenes
    outdir = BENCH / "fuzz"
    outdir.mkdir(parents=True, exist_ok=True)
    # Cache des scènes par (seed, n) : la génération coûte ~5 s/scène.
    cache = outdir / f"scenes-{seed}-{probes}.json"
    if cache.exists():
        scenes = json.loads(cache.read_text())
    else:
        print(f"  génération de {probes} scènes fuzz (seed {seed})…", flush=True)
        scenes = fuzz_scenes.generate(seed, probes)
        cache.write_text(json.dumps(scenes, indent=1))
    # Sondes Julia (plan de z, centre ≠ 0 par construction) : suite DISTINCTE,
    # ajoutée après les six familles — la classe « rebase comme si Z[0] = 0 »
    # (92-99 % de pixels faux, fermée le 2026-11-06) n'était vue par aucune
    # sonde. Une pour trois sondes, au moins une.
    n_julia = max(1, probes // 3)
    jcache = outdir / f"scenes-julia-{seed}-{n_julia}.json"
    if jcache.exists():
        scenes = scenes + json.loads(jcache.read_text())
    else:
        print(f"  génération de {n_julia} scène(s) fuzz Julia (seed {seed})…", flush=True)
        jscenes = fuzz_scenes.generate_julia(seed, n_julia)
        jcache.write_text(json.dumps(jscenes, indent=1))
        scenes = scenes + jscenes
    cases: dict[str, dict] = {}
    for s in scenes:
        name = s["name"]
        print(f"  $ fractall-quality compare --type {s['type_id']} "
              f"--zoom {s['zoom']} --name {name}", flush=True)
        jrec = journal_begin("fuzz", name, {"size": f"{width}x{height}"})
        proc = subprocess.run(
            [str(QUALITY), "compare", "--type", str(s["type_id"]),
             f"--center-x-hp={s['center_x']}",
             f"--center-y-hp={s['center_y']}",
             "--zoom", s["zoom"], "--iterations", str(s["iterations"]),
             "--width", str(width), "--height", str(height),
             "--name", name, "--output-dir", str(outdir)]
            + ([f"--julia-re={s['julia_re']}", f"--julia-im={s['julia_im']}"]
               if "julia_re" in s else []),
            cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            preexec_fn=_rlimit_preexec(mem_limit_mb))
        journal_end(jrec, _classify_rc(proc.returncode))
        verdict = "ERROR"
        rj = outdir / name / "report.json"
        if proc.returncode == 0 and rj.exists():
            try:
                verdict = json.loads(rj.read_text()).get("verdict") or "ERROR"
            except (json.JSONDecodeError, OSError):
                pass
        cases[name] = {"verdict": verdict, "type": s["type"],
                       "center_x": s["center_x"], "center_y": s["center_y"],
                       "zoom": s["zoom"], "iterations": s["iterations"]}
    verdicts = [c["verdict"] for c in cases.values()]
    return {
        "status": "ok", "seed": seed, "n_probes": len(scenes),
        "n_pass": verdicts.count("PASS"),
        "n_warn": verdicts.count("WARN"),
        "n_fail": len(verdicts) - verdicts.count("PASS") - verdicts.count("WARN"),
        "cases": cases,
        "fail_cases": [k for k, v in cases.items()
                       if v["verdict"] not in ("PASS", "WARN")],
        "warn_cases": [k for k, v in cases.items() if v["verdict"] == "WARN"],
    }
