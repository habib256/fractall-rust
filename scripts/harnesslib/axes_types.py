"""Axe `types` : TOUS les types de fractales, vitesse et correction.

Pourquoi : le corpus `toml/` et la suite quality sont des zooms Mandelbrot (et
quelques cousins). Vingt-cinq types sur trente-trois — Newton, Nova, Magnet,
Phoenix, Barnsley, Sin, Pickover, Mandelbulb, Lyapunov, densités, vectoriels,
toutes les variantes Julia — n'avaient NI chrono NI juge. L'objectif étant
l'explorateur le plus rapide et le plus juste sur tous les types, il faut un
axe qui les regarde tous.

Scènes (`harness/type-scenes.json`, figées, produites par `harness.py
types-scout`) : par type, la vue par défaut (`home`) et une vue de BORD
(`edge`, ~×4096, trouvée en zoomant quatre fois ×8 sur le voisinage le plus
contrasté) — c'est là qu'un explorateur passe son temps, pas sur la vue
d'accueil qui rend en quelques ms. Plafond d'itérations EXPLICITE et figé (la
sonde automatique est jugée par l'axe `wisdom`).

Mesures par scène :
- **vitesse** : rendu de production (Auto, CPU) à la taille de scène du tier →
  secondes, path, ns par pixel·itération ; alimente les leviers et les
  régressions vs baseline ;
- **correction** : rendu de production contre `--algorithm gmp` à PETITE
  taille (le GMP par pixel de certains types coûte ~10 µs/itération) —
  itérations brutes, seuils de l'axe truth (PASS ≤ 0,05 %, FAIL ≥ 0,5 %).
  Juge trop lent pour le budget → `judge: timeout` (n/a, pas un verdict) ;
- types SANS juge GMP (vectoriels, densités, Lyapunov) : déterminisme (deux
  rendus identiques) et non-platitude (la vue n'est pas uniforme).

**Un FAIL est toujours un vrai FAIL** : quand la production diffère du juge
(256 b), l'axe refait le juge à 512 b. Les deux juges diffèrent ⇒ la scène
est CHAOTIQUE à cette précision (frontières « hirsutes » des familles à
pliage : Buffalo, Burning Ship, Celtic, Barnsley M — 80 % de pixels ≠ entre
256 et 512 b) : verdict `CHAOS`, non comparable, aucun gap. Ils concordent ⇒
la production est fausse (premier passage, 2026-11-19 : Tricorn 2,5 %,
Barnsley Julia 5 %). Juge 512 b hors délai ⇒ FAIL « non vérifié » (sév. 4).
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import struct
import subprocess
import time
from pathlib import Path

from .axes_precision import (
    PASS_MAX_FRAC, _PATH_RE, read_fmap_iterations, verdict_for,
)
from .config import BENCH, CLI, HARNESS_DIR
from .guard import _classify_rc, _rlimit_preexec, journal_begin, journal_end

TYPE_SCENES = HARNESS_DIR / "type-scenes.json"
ALL_TYPES = list(range(1, 34))
# Pas de rendu GMP par pixel qui ait un sens : la sortie n'est pas un compte
# d'itérations d'une orbite (vectoriel, densité, exposant de Lyapunov).
NO_JUDGE = {1, 2, 16, 17, 24, 33}
JUDGE_SIZE = (64, 40)
SCOUT_SIZE = (128, 80)
SCOUT_STAGES = 4            # ×8 par étape → ×4096
# Plafond EXPLICITE des scènes jugées (éclairage compris). La sonde
# automatique n'a rien à faire ici : sur un bord elle monte à des centaines
# de milliers d'itérations (Mandelbrot ×4096 : 390 000) et s'emballe sur les
# types « fuyants » (Barnsley Mandelbrot ×5 : 2,9 M, 37 s à 128×80 — défaut
# jugé par l'axe `wisdom`). Trop lourd pour un cycle d'A/B, juge GMP compris.
SCENE_ITER = 5000
_RE_AUTO = re.compile(r"Iterations: auto → (\d+)")


def load_scenes() -> dict[str, dict]:
    """`{"<type>": {name, n, edge: {cx, cy, zoom} | None}}`."""
    if not TYPE_SCENES.exists():
        return {}
    return json.loads(TYPE_SCENES.read_text()).get("types", {})


def scene_args(tid: int, sc: dict, which: str) -> list[str] | None:
    """Arguments CLI de la scène `home` ou `edge` d'un type."""
    args = ["--type", str(tid)]
    if sc.get("n"):
        args += ["--iterations", str(sc["n"])]
    if which == "edge":
        e = sc.get("edge")
        if not e:
            return None
        args += [f"--center-x={e['cx']!r}", f"--center-y={e['cy']!r}",
                 "--zoom", repr(e["zoom"])]
    return args


def _run(args: list[str], out_map: Path, size: tuple[int, int], timeout: float,
         mem_limit_mb: int, env: dict) -> tuple[str, float | None, str]:
    cmd = [str(CLI)] + args + ["--width", str(size[0]), "--height", str(size[1]),
                               "--no-gpu",
                               "--output", str(out_map.with_suffix(".png")),
                               "--output-map", str(out_map)]
    t0 = time.monotonic()
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           env=env, timeout=timeout, text=True, errors="replace",
                           preexec_fn=_rlimit_preexec(mem_limit_mb))
    except subprocess.TimeoutExpired:
        return "timeout", None, ""
    return _classify_rc(p.returncode), time.monotonic() - t0, p.stdout or ""


def _last_path(text: str) -> str | None:
    m = None
    for m in _PATH_RE.finditer(text):
        pass
    return m.group(1) if m else None


def fmap_params(path: Path) -> dict:
    """JSON `FractalParams` d'un `.fmap` (layout `io/fmap.rs`)."""
    data = path.read_bytes()
    (n,) = struct.unpack_from("<I", data, 8)
    return json.loads(data[12:12 + n])


# --- fonctions pures (verrouillées) -----------------------------------------

def edge_pixel(it: list[int], w: int, h: int, cap: int) -> tuple[int, int]:
    """Pixel du voisinage le plus contrasté : beaucoup de comptes distincts
    dans un 5×5, médiane haute, et de préférence un mélange plafonné/évadé
    (le vrai bord, pas une équipotentielle lointaine)."""
    best = (-1.0, w // 2, h // 2)
    for y in range(3, h - 3):
        for x in range(3, w - 3):
            v = [it[(y + dy) * w + x + dx]
                 for dy in (-2, -1, 0, 1, 2) for dx in (-2, -1, 0, 1, 2)]
            f = sum(1 for q in v if q >= cap) / len(v)
            # mélange ÉQUILIBRÉ plafonné/évadé : sans ce poids l'éclaireur
            # dérive dans l'intérieur (Burning Ship : 99,9 % plafonné).
            mix = 0.3 + 4.0 * f * (1.0 - f)
            score = mix * len(set(v)) * math.log1p(statistics.median(v))
            if score > best[0]:
                best = (score, x, y)
    return best[1], best[2]


def scene_verdict(frac: float | None, deterministic: bool | None,
                  flat: bool, judge_frac: float | None = None) -> str:
    """Verdict de correction d'une scène. `judge_frac` = fraction de pixels
    qui diffèrent entre le juge à 256 b et le juge à 512 b (None = non
    mesuré) : un juge non convergé ne juge rien (`CHAOS`)."""
    if deterministic is False:
        return "FAIL"
    if frac is not None:
        if judge_frac is not None and judge_frac > PASS_MAX_FRAC:
            return "CHAOS"
        return verdict_for(frac)
    if flat:
        return "WARN"            # vue uniforme : rien n'est jugé
    return "PASS" if deterministic else "N/A"


def ns_per_px_iter(secs: float, mean_iter: float, pixels: int,
                   threads: int) -> float | None:
    """ns par (pixel × itération) et par thread — comparable entre types."""
    work = mean_iter * pixels
    return round(1e9 * secs * threads / work, 3) if work > 0 else None


# --- éclaireur --------------------------------------------------------------

def scout_type(tid: int, outdir: Path, env: dict, timeout: float = 120.0) -> dict | None:
    """Trouve la scène de bord d'un type en zoomant `SCOUT_STAGES` fois ×8 sur
    `edge_pixel`. Le plafond figé est celui que choisit la sonde automatique
    sur la vue finale (ou le défaut du type)."""
    w, h = SCOUT_SIZE
    m = outdir / f"scout_{tid}.fmap"
    fixed = [] if tid in NO_JUDGE else ["--iterations", str(SCENE_ITER)]
    st, _, text = _run(["--type", str(tid)] + fixed, m, SCOUT_SIZE, timeout,
                       0, env)
    if st != "ok" or not m.exists():
        return None
    p = fmap_params(m)
    it = read_fmap_iterations(m)
    out = {"name": p.get("fractal_type"), "n": p.get("iteration_max"),
           "edge": None}
    if tid in NO_JUDGE and tid in (1, 2):
        return out                       # vectoriels : pas de zoom utile
    cx, cy, sx, sy = p["center_x"], p["center_y"], p["span_x"], p["span_y"]
    for _ in range(SCOUT_STAGES):
        x, y = edge_pixel(it, w, h, p["iteration_max"])
        cx += (x + 0.5 - w / 2) / w * sx
        cy += (y + 0.5 - h / 2) / h * sy
        zoom = 4.0 / (sx / 8)
        st, _, text = _run(["--type", str(tid), f"--center-x={cx!r}",
                            f"--center-y={cy!r}", "--zoom", repr(zoom)]
                           + fixed, m, SCOUT_SIZE, timeout, 0, env)
        if st != "ok":
            print(f"    type {tid} : étape échouée ({st}) {text[-300:]}")
            return out
        p = fmap_params(m)
        it = read_fmap_iterations(m)
        sx, sy = p["span_x"], p["span_y"]
    out["edge"] = {"cx": cx, "cy": cy, "zoom": zoom,
                   "n": p.get("iteration_max"),
                   "distinct": len(set(it)),
                   "capped": round(sum(1 for q in it if q >= p["iteration_max"])
                                   / max(len(it), 1), 3)}
    return out


def cmd_types_scout(args) -> None:
    """Régénère `harness/type-scenes.json` (scènes FIGÉES de l'axe types).
    À relancer seulement quand un type est ajouté ou qu'une scène est jugée
    mauvaise : changer une scène casse la comparaison avec la baseline."""
    outdir = BENCH / "types"
    outdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    types = ([int(t) for t in args.types.split(",")] if args.types
             else ALL_TYPES)
    scenes = load_scenes()
    for tid in types:
        sc = scout_type(tid, outdir, env)
        if sc is None:
            print(f"  type {tid:>2} ✗ rendu échoué")
            continue
        scenes[str(tid)] = sc
        e = sc["edge"]
        print(f"  type {tid:>2} {sc['name']:30s} "
              + (f"bord ×{e['zoom']:.0f} N={e['n']} distincts={e['distinct']} "
                 f"plafonnés={e['capped']}" if e else "home seulement"))
    TYPE_SCENES.write_text(json.dumps(
        {"note": "Scènes figées de l'axe types — harness.py types-scout",
         "types": dict(sorted(scenes.items(), key=lambda kv: int(kv[0])))},
        indent=1, ensure_ascii=False) + "\n")
    print(f"✓ {TYPE_SCENES}")


# --- axe --------------------------------------------------------------------

def _scene(tid: int, sc: dict, which: str, size: tuple[int, int],
           judge_timeout: float, timeout: float, mem_limit_mb: int,
           outdir: Path, env: dict) -> dict:
    args = scene_args(tid, sc, which)
    if args is None:
        return {"status": "no_scene"}
    if which == "edge" and sc["edge"].get("n"):
        args = [a for a in args]
        i = args.index("--iterations") if "--iterations" in args else None
        if i is not None:
            args[i + 1] = str(sc["edge"]["n"])
        else:
            args += ["--iterations", str(sc["edge"]["n"])]
    stem = f"t{tid}_{which}"
    # vitesse : production à la taille du tier
    m_speed = outdir / f"{stem}.speed.fmap"
    st, secs, text = _run(args, m_speed, size, timeout, mem_limit_mb, env)
    if st != "ok":
        return {"status": st}
    it = read_fmap_iterations(m_speed)
    mean_it = sum(it) / max(len(it), 1)
    res = {"status": "ok", "type": tid, "name": sc.get("name"), "scene": which,
           "path": _last_path(text), "secs": round(secs, 4),
           "mean_iter": round(mean_it, 1),
           "ns_px_iter": ns_per_px_iter(secs, mean_it, len(it),
                                        os.cpu_count() or 1),
           "flat": len(set(it)) < 2}
    # correction : petite taille, contre GMP pur ou contre soi-même
    m_a = outdir / f"{stem}.a.fmap"
    st, _, _ = _run(args, m_a, JUDGE_SIZE, timeout, mem_limit_mb, env)
    if st != "ok":
        return {**res, "status": st}
    a = read_fmap_iterations(m_a)
    frac = det = judge_frac = None
    if tid in NO_JUDGE:
        m_b = outdir / f"{stem}.b.fmap"
        st, _, _ = _run(args, m_b, JUDGE_SIZE, timeout, mem_limit_mb, env)
        det = st == "ok" and read_fmap_iterations(m_b) == a
        res["judge"] = "determinism"
    else:
        m_g = outdir / f"{stem}.gmp.fmap"
        st, jsecs, _ = _run(args + ["--algorithm", "gmp"], m_g, JUDGE_SIZE,
                            judge_timeout, mem_limit_mb, env)
        if st == "ok":
            g = read_fmap_iterations(m_g)
            diff = sum(1 for x, y in zip(a, g) if x != y)
            frac = diff / max(len(a), 1)
            res.update({"judge": "gmp", "diff_px": diff,
                        "frac": round(frac, 6), "judge_s": round(jsecs, 2)})
            if frac > PASS_MAX_FRAC:
                # Le juge est-il convergé ? Même rendu à 512 b.
                m_g2 = outdir / f"{stem}.gmp512.fmap"
                st2, _, _ = _run(args + ["--algorithm", "gmp",
                                         "--precision-bits", "512"], m_g2,
                                 JUDGE_SIZE, judge_timeout * 3, mem_limit_mb,
                                 env)
                if st2 == "ok":
                    g2 = read_fmap_iterations(m_g2)
                    judge_frac = (sum(1 for x, y in zip(g, g2) if x != y)
                                  / max(len(g), 1))
                    res["judge_frac"] = round(judge_frac, 6)
        else:
            res["judge"] = f"gmp:{st}"
    res["verdict"] = scene_verdict(frac, det, res["flat"], judge_frac)
    return res


def axis_types(size: tuple[int, int] = (256, 160), judge_timeout: float = 20.0,
               timeout: float = 120.0, mem_limit_mb: int = 0,
               types: list[int] | None = None) -> dict:
    scenes = load_scenes()
    if not scenes:
        return {"status": "no_scenes"}
    outdir = BENCH / "types"
    outdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["FRACTALL_PERTURB_STATS"] = "1"
    cases: dict[str, dict] = {}
    todo = [(tid, which) for tid in (types or ALL_TYPES)
            if str(tid) in scenes for which in ("home", "edge")
            if which == "home" or scenes[str(tid)].get("edge")]
    for i, (tid, which) in enumerate(todo, 1):
        sc = scenes[str(tid)]
        name = f"{tid}:{sc.get('name')}:{which}"
        print(f"  [{i:>3}/{len(todo)}] types {name:44s}", end=" ", flush=True)
        jrec = journal_begin("types", name, {"size": f"{size[0]}x{size[1]}"})
        r = _scene(tid, sc, which, size, judge_timeout, timeout, mem_limit_mb,
                   outdir, env)
        journal_end(jrec, r["status"] if r["status"] in (
            "timeout", "killed_oom", "aborted", "killed") else "ok")
        cases[name] = r
        if r.get("status") != "ok":
            print(f"✗ {r.get('status')}")
            continue
        j = (f"{r['diff_px']} px ≠ GMP ({100 * r['frac']:.2f} %)"
             if r.get("frac") is not None else r.get("judge"))
        print(f"{r['verdict']:4s} {r['path'] or '—':12s} {r['secs']:.3f}s "
              f"{r['ns_px_iter']} ns/px·it — {j}")
    counts: dict[str, int] = {}
    for r in cases.values():
        if r.get("status") == "ok":
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    return {"status": "ok", "size": list(size), "judge_size": list(JUDGE_SIZE),
            "counts": counts, "cases": cases,
            "total_s": round(sum(r.get("secs") or 0 for r in cases.values()), 3)}


def types_gaps(card: dict) -> list[dict]:
    out = []
    for name, c in (card.get("types", {}).get("cases") or {}).items():
        st = c.get("status")
        if st in ("timeout", "killed_oom", "aborted", "killed", "fail"):
            out.append({"axis": "types", "case": name, "metric": "status",
                        "value": st, "baseline": None, "severity": 2,
                        "note": "rendu de production échoué ou hors délai"})
            continue
        if st != "ok" or c.get("verdict") in ("PASS", "N/A", "CHAOS"):
            continue
        v = c["verdict"]
        unverified = False
        if c.get("frac") is not None:
            unverified = c.get("judge_frac") is None
            note = (f"{c['diff_px']} px ≠ GMP pur ({100 * c['frac']:.2f} %, "
                    f"path {c.get('path')}) — "
                    + ("juge 512 b hors délai : NON vérifié" if unverified
                       else "juge convergé (256 b ≡ 512 b)"))
            metric, value = "frac", c["frac"]
        elif c.get("flat"):
            note = "vue uniforme : la scène ne juge rien (types-scout)"
            metric, value = "flat", True
        else:
            note = "deux rendus identiques diffèrent (non déterministe)"
            metric, value = "deterministic", False
        out.append({"axis": "types", "case": name, "metric": metric,
                    "value": value, "baseline": None,
                    "severity": 1 if (v == "FAIL" and not unverified) else 4,
                    "note": note})
    return out


def types_md(t: dict) -> list[str]:
    cases = t.get("cases") or {}
    cnt = t.get("counts") or {}
    L = ["- " + " · ".join(f"**{n} {k}**" for k, n in sorted(cnt.items()))
         + f" — vitesse à {t['size'][0]}×{t['size'][1]}, juge GMP à "
           f"{t['judge_size'][0]}×{t['judge_size'][1]} ; temps cumulé "
           f"**{t.get('total_s')} s**", "",
         "| Type | Scène | path | s | ns/px·it | it. moy. | juge | verdict |",
         "|---|---|---|---:|---:|---:|---|---|"]
    for name, c in cases.items():
        tid, tname, which = name.split(":")
        if c.get("status") != "ok":
            L.append(f"| {tid} {tname} | {which} | — | — | — | — | "
                     f"{c.get('status')} | — |")
            continue
        j = (f"{c['diff_px']} px ({100 * c['frac']:.2f} %)"
             if c.get("frac") is not None else c.get("judge"))
        if c.get("judge_frac") is not None:
            j += (" — juge convergé" if c["judge_frac"] <= PASS_MAX_FRAC else
                  f" — juge NON convergé ({100 * c['judge_frac']:.0f} % "
                  f"entre 256 et 512 b)")
        L.append(f"| {tid} {tname} | {which} | {c.get('path') or '—'} | "
                 f"{c['secs']:.3f} | {c.get('ns_px_iter')} | "
                 f"{c.get('mean_iter')} | {j} | {c['verdict']} |")
    L.append("")
    return L
