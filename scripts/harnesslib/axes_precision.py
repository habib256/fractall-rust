"""Axe `precision` : juge « précision ×4 » de l'orbite de référence (G14.3).

Pourquoi : la suite quality juge la perturbation contre un rendu GMP calculé
avec la MÊME formule de précision — elle ne peut pas voir une précision
d'orbite insuffisante. Le 2026-11-05, ce juge a trouvé en une passe 2-4 % de
pixels FAUX sur golden_spider / e227 / evolution_trees à la précision F3 nue
(corrigé par `ORBIT_PRECISION_MARGIN_BITS` = +64 b).

Méthode, par cas : rendu de production → `.fmap` ; rendu juge avec
`--precision-bits 4·P` (P = précision effective lue dans la ligne
`[FRACTALL] prec=…`) et `FRACTALL_ORBIT_FAST_SQR=0` (arithmétique d'orbite
indépendante : `mpc_sqr`) ; on compare les comptes d'itérations BRUTS. Le juge
est réputé convergé (vérifié ×2 == ×4 au pixel près sur les cas sensibles).
Sans dépendance (stdlib : `struct` + `zlib` pour lire le `.fmap`).
"""

from __future__ import annotations

import os
import re
import struct
import subprocess
import time
import zlib
from pathlib import Path

from .config import BENCH, CLI, TOML_DIR, cargo_build
from .guard import _classify_rc, _rlimit_preexec, journal_begin, journal_end

# Cas SENTINELLES : scènes où la précision F3 nue laissait 2-4 % de pixels
# faux (2026-11-05). Toujours mesurées en plus des cas du tier.
SENTINEL_CASES = ["golden_spider", "e227", "evolution_trees"]
JUDGE_FACTOR = 4
# Seuils (fraction de pixels dont l'itération diffère du juge). Le résidu
# attendu après correction est 0-1 px à 96² (e227 : 1 px).
PASS_MAX_FRAC = 0.0005
FAIL_MIN_FRAC = 0.005

_PREC_RE = re.compile(r"\[FRACTALL\][^\n]*?\bprec=(\d+)b")
_PATH_RE = re.compile(r"\[FRACTALL\][^\n]*?\bpath=(\S+)")


def read_fmap_iterations(path: Path) -> list[int]:
    """Canal itérations (id 0, u32 LE) d'un `.fmap` (layout `io/fmap.rs`)."""
    data = path.read_bytes()
    if data[:4] != b"FMAP":
        raise ValueError(f"{path}: magic invalide")
    off = 8
    (json_len,) = struct.unpack_from("<I", data, off)
    off += 4 + json_len
    (n_channels,) = struct.unpack_from("<I", data, off)
    off += 4
    for _ in range(n_channels):
        cid = data[off]
        raw_len, comp_len = struct.unpack_from("<QQ", data, off + 1)
        off += 17
        blob = data[off:off + comp_len]
        off += comp_len
        if cid == 0:
            raw = zlib.decompress(blob)
            if len(raw) != raw_len:
                raise ValueError(f"{path}: canal itérations tronqué")
            return list(struct.unpack(f"<{raw_len // 4}I", raw))
    raise ValueError(f"{path}: canal itérations absent")


def verdict_for(frac: float) -> str:
    if frac <= PASS_MAX_FRAC:
        return "PASS"
    if frac >= FAIL_MIN_FRAC:
        return "FAIL"
    return "WARN"


def _render(toml_path: Path, size: int, out_map: Path, extra: list[str],
            env: dict, timeout: float, mem_limit_mb: int):
    cmd = [str(CLI), "--toml", str(toml_path), "--width", str(size),
           "--height", str(size), "--no-gpu",
           "--output", str(out_map.with_suffix(".png")),
           "--output-map", str(out_map)] + extra
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          env=env, timeout=timeout, text=True, errors="replace",
                          preexec_fn=_rlimit_preexec(mem_limit_mb))


def axis_precision(cases: list[str], size: int = 96, timeout: float = 300.0,
                   mem_limit_mb: int = 0, quarantined: set | None = None,
                   slow_safe: set | None = None) -> dict:
    if not CLI.exists():
        cargo_build("fractall-cli")
    if not CLI.exists():
        return {"status": "no_binary"}
    quarantined = quarantined or set()
    slow_safe = slow_safe or set()
    outdir = BENCH / "precision"
    outdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["FRACTALL_PERTURB_STATS"] = "1"
    judge_env = dict(env, FRACTALL_ORBIT_FAST_SQR="0")
    ordered = list(dict.fromkeys(list(cases) + SENTINEL_CASES))
    results: dict[str, dict] = {}
    for i, name in enumerate(ordered, 1):
        print(f"  [{i:>3}/{len(ordered)}] precision {name:22s}", end=" ", flush=True)
        if name in quarantined or name in slow_safe:
            results[name] = {"status": "skipped"}
            print("⊘ skip")
            continue
        toml_path = TOML_DIR / f"{name}.toml"
        if not toml_path.exists():
            results[name] = {"status": "missing_toml"}
            print("✗ toml absent")
            continue
        jrec = journal_begin("precision", name, {"size": f"{size}x{size}"})
        t0 = time.monotonic()
        prod_map = outdir / f"{name}.prod.fmap"
        try:
            proc = _render(toml_path, size, prod_map, [], env, timeout, mem_limit_mb)
        except subprocess.TimeoutExpired:
            journal_end(jrec, "timeout")
            results[name] = {"status": "timeout"}
            print("⏱ timeout")
            continue
        m = _PREC_RE.search(proc.stdout or "")
        if _classify_rc(proc.returncode) != "ok" or m is None:
            # Pas de ligne prec= : path sans orbite GMP (f64 standard, densité…).
            journal_end(jrec, _classify_rc(proc.returncode))
            results[name] = {"status": "n/a" if proc.returncode == 0 else "error"}
            print("— sans orbite GMP" if proc.returncode == 0 else "✗ erreur")
            continue
        path_m = _PATH_RE.search(proc.stdout or "")
        if path_m and path_m.group(1).startswith("std"):
            # Path f64 standard : aucune orbite GMP, `--precision-bits` sans
            # effet — un « PASS » y serait vide de sens.
            journal_end(jrec, "ok")
            results[name] = {"status": "n/a", "path": path_m.group(1)}
            print(f"— sans orbite GMP ({path_m.group(1)})")
            continue
        prec = int(m.group(1))
        judge_map = outdir / f"{name}.judge.fmap"
        try:
            jproc = _render(toml_path, size, judge_map,
                            ["--precision-bits", str(prec * JUDGE_FACTOR)],
                            judge_env, timeout * 4, mem_limit_mb)
        except subprocess.TimeoutExpired:
            journal_end(jrec, "timeout")
            results[name] = {"status": "timeout", "prec": prec}
            print("⏱ timeout (juge)")
            continue
        status = _classify_rc(jproc.returncode)
        journal_end(jrec, status, {"secs": round(time.monotonic() - t0, 3)})
        if status != "ok":
            results[name] = {"status": status, "prec": prec}
            print(f"✗ juge {status}")
            continue
        a = read_fmap_iterations(prod_map)
        b = read_fmap_iterations(judge_map)
        diff = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
        frac = diff / max(len(a), 1)
        verdict = verdict_for(frac)
        results[name] = {"status": "ok", "prec": prec,
                         "judge_prec": prec * JUDGE_FACTOR, "diff_px": diff,
                         "frac": round(frac, 6), "verdict": verdict,
                         "sentinel": name in SENTINEL_CASES}
        print(f"prec={prec}b juge={prec * JUDGE_FACTOR}b diff={diff} px "
              f"({100 * frac:.3f} %) {verdict}")
    ok = [r for r in results.values() if r.get("status") == "ok"]
    verdicts = [r["verdict"] for r in ok]
    return {
        "status": "ok", "size": size, "judge_factor": JUDGE_FACTOR,
        "n_cases": len(ok), "n_pass": verdicts.count("PASS"),
        "n_warn": verdicts.count("WARN"), "n_fail": verdicts.count("FAIL"),
        "cases": results,
    }
