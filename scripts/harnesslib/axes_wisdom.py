"""Axe `wisdom` : juge les DÉCISIONS automatiques du moteur (G9).

Pourquoi : tous les autres axes jugent un rendu (juste ? rapide ?), aucun ne
juge le CHOIX qui l'a produit. Un seuil de routage mal calibré, ou devenu faux
après qu'un chemin a été accéléré, ne casse rien : il rend l'image juste, plus
lentement qu'il ne pourrait — invisible sans mesure du contrefactuel.

Deux volets, mêmes sorties (`cases` nom → dict, verdict PASS/WARN/FAIL/…) :

1. **Regret de routage** (`decision:<dimension>:<cas>`) : le plan choisi
   (`auto`) est chronométré contre chaque alternative FORCÉE de la dimension
   (variable d'environnement ou option CLI). Le regret ne compte qu'à SORTIE
   ÉQUIVALENTE — itérations brutes des `.fmap` égales à `EQUIV_MAX_FRAC` près.
   Une alternative plus rapide qui rend autre chose est un compromis de
   correction, pas un gap de vitesse : verdict `ADJUDICATE` (à trancher contre
   le GMP pur, puis à persister dans `harness/wisdom-adjudications.json`).
2. **Itérations automatiques** (`iter:<scène>`) : la sonde `wisdom_iter`
   choisit un plafond N. Juge : le même rendu à 4N. Sous-itération = fraction
   de pixels plafonnés à N qui s'évadent avant 4N (le critère du moteur lui-même :
   cible 0,2 %, croissance au-delà de 0,5 %). Sur-provision = plus petit
   N/4^k qui satisfait encore le critère, et ce qu'il aurait coûté en temps.

Sans dépendance (stdlib). CPU seulement (`--no-gpu`).
"""

from __future__ import annotations

import os
import re
import statistics
import subprocess
import time
from pathlib import Path

from .axes_precision import (
    FAIL_MIN_FRAC, PASS_MAX_FRAC, _PATH_RE, read_fmap_iterations,
)
from .config import BENCH, CLI, TOML_DIR
from .guard import _classify_rc, _rlimit_preexec, journal_begin, journal_end

# --- seuils -----------------------------------------------------------------
REGRET_WARN = 1.10          # auto / meilleure alternative équivalente
REGRET_FAIL = 1.25
NOISE_FLOOR_S = 0.060       # sous cet écart absolu, le chrono n'est pas fiable
EQUIV_MAX_FRAC = 0.0005     # sorties « équivalentes » (seuil PASS des axes truth)
ITER_UNDER_PASS = 0.005     # = GROW_THRESHOLD du moteur (wisdom_iter)
ITER_UNDER_FAIL = 0.02
ITER_TARGET = 0.002         # = TARGET_CHANGE du moteur
ITER_WASTE_WARN = 16        # le moteur ne réduit que si C/4 suffit : 4× est voulu
ITER_WASTE_TIME_WARN = 1.25

WIDTH, HEIGHT = 256, 160

SEAHORSE = ("-0.743643887037158704752191506114774",
            "0.131825904205311970493132056385139")
ELEPHANTS = ("0.27631692275258796", "0.0085")
MINIBROT = ("-1.7548776662466927", "0.0")


def _view(ftype: int, center: tuple[str, str], zoom: str) -> list[str]:
    return ["--type", str(ftype), f"--center-x-hp={center[0]}",
            f"--center-y-hp={center[1]}", "--zoom", zoom]


# Dimensions de décision. `variants` : nom → (env, options CLI) ; `auto` est
# le plan du wisdom. `cases` : nom → arguments de scène (stem toml/ ou vue en
# ligne à plafond EXPLICITE — sinon la sonde d'itérations entrerait dans le
# chrono). (quick, full) = échantillons par tier.
DIMENSIONS: dict[str, dict] = {
    # Harmonic LA vs BLA (G9.3, seuil `route_harmonic_auto` period0 ≤ 100) :
    # périodes courtes (LLA gagne) ET longues (BLA gagne).
    "harmonic": {
        "variants": {
            "auto": ({"FRACTALL_HARMONIC_LA": "auto"}, []),
            "bla": ({"FRACTALL_HARMONIC_LA": "bla"}, []),
            "lla": ({"FRACTALL_HARMONIC_LA": "lla"}, []),
        },
        "quick": ["flake", "mitosis", "dinosaur_fossils", "e50", "e113"],
        "full": ["flake", "test3", "mitosis", "glitch_test_5",
                 "dinosaur_fossils", "e50", "e113", "dragon"],
    },
    # f64 standard (tracé + corrigé) vs perturbation : de part et d'autre de
    # la bascule Auto (~1e12-1e13 à cette taille). Le path standard a gagné
    # un traqueur et une correction le 2026-11-18 : son coût réel a bougé.
    "algorithm": {
        "variants": {
            "auto": ({}, []),
            "f64": ({}, ["--algorithm", "f64"]),
            "perturbation": ({}, ["--algorithm", "perturbation"]),
        },
        "scenes": {
            "seahorse_1e7": _view(3, SEAHORSE, "1e7") + ["--iterations", "2500"],
            "seahorse_1e10": _view(3, SEAHORSE, "1e10") + ["--iterations", "2500"],
            "seahorse_1e12": _view(3, SEAHORSE, "1e12") + ["--iterations", "5000"],
            "elephants_1e10": _view(3, ELEPHANTS, "1e10") + ["--iterations", "2500"],
            "minibrot_1e10": _view(3, MINIBROT, "1e10") + ["--iterations", "2500"],
            "elephants_1e7": _view(3, ELEPHANTS, "1e7") + ["--iterations", "2500"],
            # Plafond que choisit la sonde d'itérations sur cette vue : le
            # régime RÉEL d'un utilisateur en « Iter: Auto » (bord parabolique,
            # 43 % de pixels signalés par le traqueur du path standard).
            "elephants_1e7_60k": _view(3, ELEPHANTS, "1e7")
            + ["--iterations", "60000"],
            "seahorse_1e7_90k": _view(3, SEAHORSE, "1e7")
            + ["--iterations", "90000"],
        },
        "quick": ["seahorse_1e7", "seahorse_1e10", "seahorse_1e12",
                  "elephants_1e10"],
        "full": ["seahorse_1e7", "seahorse_1e10", "seahorse_1e12",
                 "elephants_1e10", "minibrot_1e10", "elephants_1e7",
                 "elephants_1e7_60k", "seahorse_1e7_90k"],
    },
}

# Scènes du volet itérations automatiques (aucun `--iterations` : la sonde
# décide). (quick, full).
ITER_SCENES: dict[str, list[str]] = {
    "mandelbrot_home": ["--type", "3"],
    "julia_home": ["--type", "4"],
    "seahorse_1e4": _view(3, SEAHORSE, "1e4"),
    "seahorse_1e7": _view(3, SEAHORSE, "1e7"),
    "elephants_1e7": _view(3, ELEPHANTS, "1e7"),
    "minibrot_1e7": _view(3, MINIBROT, "1e7"),
    "burning_ship_home": ["--type", "13"],
    # Types « fuyants » : l'intérieur apparent finit par s'évader, la queue de
    # la distribution ne s'éteint pas et la sonde court après (2026-11-19 :
    # Barnsley Mandelbrot → 2,9 M itérations, 37 s à 128×80 ; Multibrot à
    # puissance non entière → 140 000).
    "barnsley_m_zoom": ["--type", "10", "--center-x=-0.4921875",
                        "--center-y=-1.225", "--zoom", "5.333333333333333"],
    "multibrot_zoom": ["--type", "23", "--center-x=-0.796875",
                       "--center-y=-0.16875", "--zoom", "8"],
    "seahorse_1e10": _view(3, SEAHORSE, "1e10"),
}
ITER_QUICK = ["mandelbrot_home", "julia_home", "seahorse_1e4", "seahorse_1e7",
              "elephants_1e7", "barnsley_m_zoom", "multibrot_zoom"]
# Budget d'UNE scène du volet itérations (sonde + rendu) : au-delà, un
# explorateur en « Iter: Auto » est inutilisable sur cette vue → gap.
ITER_BUDGET_S = 20.0
ITER_FULL = list(ITER_SCENES)

_RE_AUTO = re.compile(r"Iterations: auto → (\d+) \((.*?)\)")
_RE_PROBE_S = re.compile(r"([\d.]+) s\s*$")
_RE_DONE = re.compile(r"Done in ([\d.]+)s \(render: ([\d.]+)s")
_RE_FIAB_STD = re.compile(
    r"\[RELIABILITY-STD\] flagged=\d+ \(([\d.]+) %\).*?time=([\d.]+)s")
_RE_FIAB_PERT = re.compile(
    r"\[RELIABILITY\][^\n]*?flagged=\d+/\d+ \(([\d.]+)%\)")
_RE_POST = re.compile(r"\[PERTURB PERF\][^\n]*?\bpost=([\d.]+)s")


# --- fonctions pures (verrouillées par test_harness_guard) --------------------

def diff_frac(a: list[int], b: list[int]) -> float:
    """Fraction de pixels dont l'itération brute diffère."""
    n = max(len(a), len(b), 1)
    return (sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))) / n


def regret_verdict(chosen_s: float, alts: dict[str, dict]) -> dict:
    """Verdict de regret. `alts` : variante → {secs, frac} (frac = écart de
    sortie avec `auto`). Seules les alternatives ÉQUIVALENTES font un regret
    de vitesse ; une alternative plus rapide à sortie ≠ donne ADJUDICATE."""
    equiv = {v: a for v, a in alts.items() if a["frac"] <= EQUIV_MAX_FRAC}
    other = {v: a for v, a in alts.items() if a["frac"] > EQUIV_MAX_FRAC}

    def best(d):
        if not d:
            return None, None
        v = min(d, key=lambda k: d[k]["secs"])
        return v, d[v]["secs"]

    def beats(secs):          # l'alternative bat-elle `auto` hors du bruit ?
        return (secs is not None and chosen_s / secs > REGRET_WARN
                and chosen_s - secs >= NOISE_FLOOR_S)

    ev, es = best(equiv)
    ov, osecs = best(other)
    out = {"best_equiv": ev, "best_equiv_s": es, "best_other": ov,
           "best_other_s": osecs,
           "regret": round(chosen_s / es, 3) if es else None,
           "regret_s": round(chosen_s - es, 4) if es else None}
    if beats(es):
        out["verdict"] = "FAIL" if chosen_s / es > REGRET_FAIL else "WARN"
    elif beats(osecs):
        out["verdict"] = "ADJUDICATE"
        out["regret_other"] = round(chosen_s / osecs, 3)
    else:
        out["verdict"] = "PASS"
    return out


def adjudged_verdict(res: dict, frac_auto: float, frac_alt: float) -> dict:
    """Tranche un ADJUDICATE contre le GMP pur. `frac_*` = fraction de pixels
    ≠ juge. Ordre : (1) le rendu de PRODUCTION est faux → FAIL de correction
    (`wrong`), quelle que soit l'alternative ; (2) l'alternative rapide est
    juste elle aussi → vrai regret de vitesse ; (3) elle est fausse → le
    wisdom a raison de payer plus cher (PASS, `alt_wrong`)."""
    out = dict(res)
    out["judge"] = {"algo": "gmp", "frac_auto": round(frac_auto, 6),
                    "frac_alt": round(frac_alt, 6)}
    if frac_auto > PASS_MAX_FRAC:
        out["verdict"] = "FAIL" if frac_auto >= FAIL_MIN_FRAC else "WARN"
        out["adjudged"] = "auto_wrong"
    elif frac_alt <= PASS_MAX_FRAC:
        ratio = res["chosen_s"] / res["best_other_s"]
        out["verdict"] = "FAIL" if ratio > REGRET_FAIL else "WARN"
        out["adjudged"] = "alt_right"
        out.update({"best_equiv": res["best_other"],
                    "best_equiv_s": res["best_other_s"],
                    "regret": round(ratio, 3),
                    "regret_s": round(res["chosen_s"] - res["best_other_s"], 4)})
    else:
        out["verdict"] = "PASS"
        out["adjudged"] = "alt_wrong"
    return out


def under_iteration_frac(at_n: list[int], at_4n: list[int], n: int) -> float:
    """Pixels plafonnés à N qui s'évadent avant 4N (classe qui change)."""
    cap4 = 4 * n
    tot = max(len(at_n), 1)
    return sum(1 for a, b in zip(at_n, at_4n) if a >= n and b < cap4) / tot


def smallest_sufficient(at_4n: list[int], n: int, floor: int = 16) -> int:
    """Plus petit plafond C = N/4^k qui satisfait ENCORE le critère du moteur
    (moins de `ITER_TARGET` des pixels changent de classe entre C et 4C),
    en descendant tant que le critère tient. Rend N si N/4 ne suffit pas."""
    tot = max(len(at_4n), 1)
    best = n
    c = n // 4
    while c >= floor:
        changing = sum(1 for b in at_4n if c <= b < 4 * c) / tot
        if changing > ITER_TARGET:
            break
        best = c
        c //= 4
    return best


def iter_verdict(under: float, waste: int, time_ratio: float | None,
                 time_delta_s: float | None = None) -> str:
    if under > ITER_UNDER_FAIL:
        return "FAIL"
    if under > ITER_UNDER_PASS:
        return "WARN"
    if (waste >= ITER_WASTE_WARN and time_ratio is not None
            and time_ratio > ITER_WASTE_TIME_WARN
            and (time_delta_s is None or time_delta_s >= NOISE_FLOOR_S)):
        return "WASTE"
    return "PASS"


def parse_auto_line(text: str) -> tuple[int | None, float | None]:
    """(N choisi, secondes de sonde) depuis la ligne « Iterations: auto → N »."""
    m = _RE_AUTO.search(text)
    if not m:
        return None, None
    ps = _RE_PROBE_S.search(m.group(2))
    return int(m.group(1)), (float(ps.group(1)) if ps else None)


def parse_reliability(text: str) -> tuple[float | None, float | None]:
    """(% de pixels signalés, secondes de correction) du DERNIER rendu du
    journal (la sonde d'itérations en imprime aussi). Path standard : ligne
    `[RELIABILITY-STD]` ; perturbation : `[RELIABILITY]` + `post=`."""
    std = _RE_FIAB_STD.findall(text)
    if std:
        return float(std[-1][0]), float(std[-1][1])
    pert, post = _RE_FIAB_PERT.findall(text), _RE_POST.findall(text)
    if pert:
        return float(pert[-1]), (float(post[-1]) if post else None)
    return None, None


def parse_render_s(text: str) -> float | None:
    m = _RE_DONE.search(text)
    return float(m.group(2)) if m else None


# --- exécution --------------------------------------------------------------

def _run(args: list[str], out_map: Path, env: dict, timeout: float,
         mem_limit_mb: int,
         size: tuple[int, int] = (WIDTH, HEIGHT)) -> tuple[str, float | None, str]:
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


def _scene_args(dim: dict, case: str) -> list[str] | None:
    if "scenes" in dim:
        return list(dim["scenes"][case])
    toml_path = TOML_DIR / f"{case}.toml"
    if not toml_path.exists():
        return None
    return ["--toml", str(toml_path), "--bailout", "25"]


def _decision_case(dname: str, dim: dict, case: str, runs: int, timeout: float,
                   mem_limit_mb: int, outdir: Path, base_env: dict,
                   adjudications: dict,
                   size: tuple[int, int] = (WIDTH, HEIGHT)) -> dict:
    args = _scene_args(dim, case)
    if args is None:
        return {"status": "toml_absent"}
    variants = list(dim["variants"])
    times: dict[str, list[float]] = {v: [] for v in variants}
    maps: dict[str, Path] = {}
    path = None
    # Passes ENTRELACÉES (auto, a, b, auto, a, b, …) : une dérive thermique ou
    # une charge passagère touche toutes les variantes, pas une seule.
    for _ in range(runs):
        for v in variants:
            env_over, flags = dim["variants"][v]
            env = dict(base_env)
            env.update(env_over)
            maps[v] = outdir / f"{dname}_{case}_{v}.fmap"
            st, secs, text = _run(args + flags, maps[v], env, timeout,
                                  mem_limit_mb, size)
            if st != "ok":
                return {"status": st, "variant": v}
            times[v].append(secs)
            if v == "auto":
                path = _last_path(text) or path
    med = {v: statistics.median(t) for v, t in times.items()}
    ref = read_fmap_iterations(maps["auto"])
    alts = {v: {"secs": med[v],
                "frac": diff_frac(ref, read_fmap_iterations(maps[v]))}
            for v in variants if v != "auto"}
    res = regret_verdict(med["auto"], alts)
    res.update({
        "status": "ok", "kind": "decision", "dimension": dname, "path": path,
        "chosen_s": round(med["auto"], 4),
        "times": {v: round(s, 4) for v, s in med.items()},
        "out_frac": {v: round(a["frac"], 6) for v, a in alts.items()},
    })
    for k in ("best_equiv_s", "best_other_s"):
        if res.get(k) is not None:
            res[k] = round(res[k], 4)
    if res["verdict"] == "ADJUDICATE":
        adj = adjudications.get(case)
        if adj is not None:
            same = (adj.get("routed_path") == path
                    and adj.get("faster_alt") == res["best_other"])
            res["adj_status"] = "known" if same else "regime_changed"
        elif "scenes" in dim:
            # Vue peu profonde : le GMP pur par pixel est abordable, on
            # tranche tout de suite (les cas toml/ profonds passent par
            # `harness/wisdom-adjudications.json`).
            judge_map = outdir / f"{dname}_{case}_gmp.fmap"
            st, secs, _ = _run(args + ["--algorithm", "gmp"], judge_map,
                               base_env, timeout * 4, mem_limit_mb, size)
            if st == "ok":
                g = read_fmap_iterations(judge_map)
                res = adjudged_verdict(
                    res, diff_frac(ref, g),
                    diff_frac(read_fmap_iterations(maps[res["best_other"]]), g))
                res["judge"]["secs"] = round(secs, 2)
            else:
                res["judge"] = {"algo": "gmp", "status": st}
    return res


def _iter_case(name: str, timeout: float, mem_limit_mb: int, outdir: Path,
               base_env: dict, size: tuple[int, int] = (WIDTH, HEIGHT)) -> dict:
    scene = ITER_SCENES[name]
    m_auto = outdir / f"iter_{name}_auto.fmap"
    st, wall, text = _run(scene, m_auto, base_env,
                          min(timeout, ITER_BUDGET_S), mem_limit_mb, size)
    if st != "ok":
        return {"status": st, "kind": "iter", "budget_s": ITER_BUDGET_S}
    n, probe_s = parse_auto_line(text)
    if n is None:
        return {"status": "no_auto_line"}
    render_s = parse_render_s(text)
    fiab_pct, fiab_s = parse_reliability(text)
    m_4n = outdir / f"iter_{name}_4n.fmap"
    st, _, _ = _run(scene + ["--iterations", str(4 * n)], m_4n, base_env,
                    timeout * 4, mem_limit_mb, size)
    if st != "ok":
        return {"status": st, "auto_iterations": n}
    a, b = read_fmap_iterations(m_auto), read_fmap_iterations(m_4n)
    under = under_iteration_frac(a, b, n)
    c_ok = smallest_sufficient(b, n)
    waste = n // max(c_ok, 1)
    res = {"status": "ok", "kind": "iter", "auto_iterations": n,
           "under_frac": round(under, 6), "sufficient": c_ok, "waste": waste,
           "probe_s": probe_s, "render_s": render_s,
           "path": _last_path(text), "fiab_pct": fiab_pct, "fiab_s": fiab_s,
           "wall_s": round(wall, 4),
           "capped_frac": round(sum(1 for x in a if x >= n) / max(len(a), 1), 4),
           "no_escape": "no escape" in text.lower() or "no_escape" in text}
    time_ratio = delta = None
    if waste >= 4:
        # Ce que le plafond suffisant aurait coûté : même rendu à C, à N.
        m_c = outdir / f"iter_{name}_c.fmap"
        st_n, t_n, _ = _run(scene + ["--iterations", str(n)], m_c, base_env,
                            timeout, mem_limit_mb, size)
        st_c, t_c, _ = _run(scene + ["--iterations", str(c_ok)], m_c, base_env,
                            timeout, mem_limit_mb, size)
        if st_n == "ok" and st_c == "ok" and t_c > 0:
            time_ratio, delta = t_n / t_c, t_n - t_c
            res.update({"time_at_n_s": round(t_n, 4),
                        "time_at_sufficient_s": round(t_c, 4),
                        "waste_time_ratio": round(time_ratio, 3)})
    res["verdict"] = iter_verdict(under, waste, time_ratio, delta)
    return res


def axis_wisdom(full: bool, runs: int = 3, timeout: float = 300.0,
                mem_limit_mb: int = 0, adjudications: dict | None = None,
                only: str | None = None,
                size: tuple[int, int] = (WIDTH, HEIGHT)) -> dict:
    """Exécute les deux volets. `only` = `decision` | `iter` | nom de
    dimension, pour un cycle ciblé."""
    outdir = BENCH / "wisdom"
    outdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["FRACTALL_PERTURB_STATS"] = "1"
    adjudications = adjudications or {}
    cases: dict[str, dict] = {}
    todo: list[tuple[str, str, str]] = []
    for dname, dim in DIMENSIONS.items():
        if only in (None, "decision", dname):
            todo += [("decision", dname, c)
                     for c in dim["full" if full else "quick"]]
    if only in (None, "iter"):
        todo += [("iter", "", s) for s in (ITER_FULL if full else ITER_QUICK)]
    for i, (kind, dname, case) in enumerate(todo, 1):
        name = f"decision:{dname}:{case}" if kind == "decision" else f"iter:{case}"
        print(f"  [{i:>3}/{len(todo)}] wisdom {name:38s}", end=" ", flush=True)
        jrec = journal_begin("wisdom", name, {"size": f"{size[0]}x{size[1]}"})
        if kind == "decision":
            r = _decision_case(dname, DIMENSIONS[dname], case, runs, timeout,
                               mem_limit_mb, outdir, env, adjudications, size)
        else:
            r = _iter_case(case, timeout, mem_limit_mb, outdir, env, size)
        journal_end(jrec, r["status"] if r["status"] in (
            "timeout", "killed_oom", "aborted", "killed") else "ok")
        cases[name] = r
        print(_line(r))
    counts: dict[str, int] = {}
    for r in cases.values():
        if r.get("status") == "ok":
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    return {"status": "ok", "runs": runs, "size": list(size),
            "counts": counts, "cases": cases,
            "thresholds": {"regret_warn": REGRET_WARN,
                           "regret_fail": REGRET_FAIL,
                           "equiv_max_frac": EQUIV_MAX_FRAC,
                           "iter_under_pass": ITER_UNDER_PASS,
                           "iter_under_fail": ITER_UNDER_FAIL}}


def _line(r: dict) -> str:
    if r.get("status") != "ok":
        return f"✗ {r.get('status')}"
    if r["kind"] == "decision":
        alt = (f"{r['best_equiv']} {r['best_equiv_s']:.3f}s ×{r['regret']}"
               if r.get("best_equiv") else "aucune alternative équivalente")
        tag = ""
        if r.get("adjudged"):
            j = r["judge"]
            tag = (f" [adjugé GMP : auto {100 * j['frac_auto']:.2f} % faux, "
                   f"`{r['best_other']}` {r['best_other_s']:.3f}s "
                   f"{100 * j['frac_alt']:.2f} % faux]")
        elif r["verdict"] == "ADJUDICATE":
            tag = (f" [{r['best_other']} {r['best_other_s']:.3f}s sortie≠"
                   f"{' déjà-adjugé' if r.get('adj_status') == 'known' else ''}]")
        return (f"{r['verdict']} {r['path']} auto={r['chosen_s']:.3f}s "
                f"vs {alt}{tag}")
    extra = (f" (N/{r['waste']} suffisait"
             + (f", ×{r['waste_time_ratio']} en temps"
                if r.get("waste_time_ratio") else "") + ")"
             if r["waste"] > 1 else "")
    return (f"{r['verdict']} N={r['auto_iterations']} "
            f"sous-itération {100 * r['under_frac']:.3f} %{extra} "
            f"sonde {r['probe_s']} s / rendu {r['render_s']} s")


def wisdom_gaps(card: dict, adjudications: dict) -> list[dict]:
    """Gaps de l'axe : FAIL de regret → vitesse (3) ; sous-itération FAIL →
    correction (1) ; WARN / WASTE / ADJUDICATE non tranché → qualité (4)."""
    out = []
    for name, c in (card.get("wisdom", {}).get("cases") or {}).items():
        if c.get("status") == "timeout" and c.get("kind") == "iter":
            out.append({"axis": "wisdom", "case": name, "metric": "status",
                        "value": "timeout", "baseline": None, "severity": 2,
                        "note": f"itérations auto : sonde + rendu > "
                                f"{c.get('budget_s')} s à petite taille — le "
                                f"plafond choisi s'emballe (type fuyant ?)"})
            continue
        if c.get("status") != "ok" or c.get("verdict") == "PASS":
            continue
        v = c["verdict"]
        if c["kind"] == "decision":
            if c.get("adjudged") == "auto_wrong":
                fa = c["judge"]["frac_auto"]
                out.append({"axis": "wisdom", "case": name,
                            "metric": "wrong_frac", "value": fa,
                            "baseline": None,
                            "severity": 1 if v == "FAIL" else 4,
                            "note": f"le rendu de PRODUCTION ({c['path']}) a "
                                    f"{100 * fa:.2f} % de pixels ≠ GMP pur "
                                    f"(alternative `{c['best_other']}` : "
                                    f"{100 * c['judge']['frac_alt']:.2f} %)"})
            elif v == "ADJUDICATE":
                if c.get("adj_status") == "known":
                    continue
                note = (f"`{c['best_other']}` ×{c.get('regret_other')} plus "
                        f"rapide mais sortie ≠ "
                        f"({100 * c['out_frac'][c['best_other']]:.2f} % px)"
                        + (" — RÉGIME CHANGÉ depuis l'adjudication"
                           if c.get("adj_status") == "regime_changed" else "")
                        + " — trancher vs GMP pur")
                out.append({"axis": "wisdom", "case": name,
                            "metric": "regret_other",
                            "value": c.get("regret_other"), "baseline": None,
                            "severity": 4, "note": note})
            else:
                out.append({"axis": "wisdom", "case": name, "metric": "regret",
                            "value": c["regret"], "baseline": None,
                            "severity": 3 if v == "FAIL" else 4,
                            "note": f"le plan auto ({c['path']}) est "
                                    f"×{c['regret']} plus lent que "
                                    f"`{c['best_equiv']}` à sortie équivalente "
                                    f"(+{c['regret_s']:.2f} s)"})
        elif v in ("FAIL", "WARN"):
            out.append({"axis": "wisdom", "case": name, "metric": "under_frac",
                        "value": c["under_frac"], "baseline": None,
                        "severity": 1 if v == "FAIL" else 4,
                        "note": f"itérations auto N={c['auto_iterations']} : "
                                f"{100 * c['under_frac']:.2f} % de pixels "
                                f"plafonnés s'évadent avant 4N"})
        elif v == "WASTE":
            out.append({"axis": "wisdom", "case": name,
                        "metric": "waste_time_ratio",
                        "value": c.get("waste_time_ratio"), "baseline": None,
                        "severity": 4,
                        "note": f"itérations auto N={c['auto_iterations']} : "
                                f"N/{c['waste']} suffisait, "
                                f"×{c.get('waste_time_ratio')} en temps"})
    return out


def wisdom_md(w: dict) -> list[str]:
    """Section du scorecard."""
    cases = w.get("cases") or {}
    cnt = w.get("counts") or {}
    L = ["- " + " · ".join(f"**{n} {k}**" for k, n in sorted(cnt.items()))
         + f" — {w.get('size', ['?', '?'])[0]}×{w.get('size', ['?', '?'])[1]}"
           f", {w.get('runs')} passes entrelacées (médiane)", ""]
    dec = {n: c for n, c in cases.items() if c.get("kind") == "decision"}
    if dec:
        L += ["**Regret de routage** — `auto` contre chaque plan forcé ; le "
              "regret ne compte qu'à sortie équivalente.", "",
              "| Décision | path auto | auto (s) | meilleure équivalente | "
              "regret | plus rapide à sortie ≠ | verdict |",
              "|---|---|---:|---|---:|---|---|"]
        for n, c in dec.items():
            eq = (f"`{c['best_equiv']}` {c['best_equiv_s']:.3f} s"
                  if c.get("best_equiv") else "—")
            ot = "—"
            if c.get("adjudged"):
                j = c["judge"]
                ot = (f"`{c['best_other']}` {c['best_other_s']:.3f} s — adjugé "
                      f"GMP : auto {100 * j['frac_auto']:.2f} % faux, "
                      f"alt. {100 * j['frac_alt']:.2f} % faux")
            elif c["verdict"] == "ADJUDICATE":
                ot = (f"`{c['best_other']}` {c['best_other_s']:.3f} s"
                      + (" (adjugé ✓)" if c.get("adj_status") == "known"
                         else " (⚠ régime changé)"
                         if c.get("adj_status") == "regime_changed" else ""))
            L.append(f"| `{n.split(':', 1)[1]}` | {c.get('path')} | "
                     f"{c['chosen_s']:.3f} | {eq} | "
                     f"{c['regret'] if c.get('regret') else '—'} | {ot} | "
                     f"{c['verdict']} |")
        L.append("")
    # Une scène hors budget (timeout) porte `kind: iter` sans mesure : elle
    # est listée sous « Non mesurés », pas dans le tableau.
    it = {n: c for n, c in cases.items()
          if c.get("kind") == "iter" and c.get("status") == "ok"}
    if it:
        L += ["**Itérations automatiques** — juge : le même rendu à 4N.", "",
              "| Scène | N auto | sous-itération % | suffisant | "
              "surcoût temps | sonde (s) | rendu (s) | verdict |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
        for n, c in it.items():
            wt = (f"×{c['waste_time_ratio']}" if c.get("waste_time_ratio")
                  else "—")
            L.append(f"| `{n.split(':', 1)[1]}` | {c['auto_iterations']} | "
                     f"{100 * c['under_frac']:.3f} | {c['sufficient']} | {wt} | "
                     f"{c.get('probe_s')} | {c.get('render_s')} | "
                     f"{c['verdict']} |")
        L.append("")
    bad = {n: c for n, c in cases.items() if c.get("status") != "ok"}
    if bad:
        L += ["_Non mesurés : " + ", ".join(
            f"`{n}` ({c.get('status')})" for n, c in bad.items()) + "_", ""]
    return L
