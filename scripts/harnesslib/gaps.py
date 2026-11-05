"""Calcul des gaps (liste de travail triée) et régressions vs baseline."""

from __future__ import annotations

import json
import math

from .config import ADJUDICATIONS, WISDOM_ADJUDICATIONS

def load_adjudications() -> dict:
    """Verdicts 3-voies persistés (harness adjudicate) : case → qui avait tort
    (fractall/F3/partagé) vs GMP ground truth. Consommés par compute_gaps pour
    annoter les divergences parité — fin du re-litige manuel."""
    if ADJUDICATIONS.exists():
        try:
            return json.loads(ADJUDICATIONS.read_text())
        except json.JSONDecodeError:
            return {}
    return {}

def load_wisdom_adjudications() -> dict:
    """Tradeoffs correction déjà tranchés de l'axe wisdom-optimality : case →
    {routed_path, faster_alt, reason, ...}. Consommés par cmd_wisdom_opt pour
    marquer un ADJUDICATE connu « déjà-adjugé✓ » (le wisdom route le path
    lent-mais-CORRECT ; l'alternative rapide est fausse vs GMP). Le verdict brut
    reste ADJUDICATE ; seul le régime (routed_path + output_differs) est comparé
    à l'enregistrement — un changement ré-escalade en `⚠ RÉGIME CHANGÉ`."""
    if WISDOM_ADJUDICATIONS.exists():
        try:
            return json.loads(WISDOM_ADJUDICATIONS.read_text())
        except json.JSONDecodeError:
            return {}
    return {}

def _gap(axis, case, metric, value, baseline_value, severity, note):
    return {"axis": axis, "case": case, "metric": metric, "value": value,
            "baseline_value": baseline_value, "severity": severity,
            "note": note}

def _comparable(card: dict, base: dict) -> bool:
    return (base.get("meta", {}).get("tier") == card["meta"]["tier"])

def _same_machine(card: dict, base: dict) -> bool:
    """Baseline par machine (cf. HARNESS.md) : les métriques de VITESSE (ratios
    fractall/F3, geomean) ne sont comparables qu'entre runs de la MÊME machine —
    le nombre de cœurs et le CPU changent l'équilibre orbite-série (F3 et fractall
    ne scalent pas identiquement). Sans ce garde-fou, rejouer le harness sur une
    autre machine que la baseline flagge une fausse « RÉGRESSION geomean » (ex.
    baseline i7-10700F 16 threads vs Xeon 4 threads → geomean 0.33→0.63 alarme à
    tort). Les axes de CORRECTION (parité pixel-equiv, quality FAIL, goldens) sont
    machine-indépendants → non gatés."""
    cm = card.get("meta", {}).get("machine", {}) or {}
    bm = base.get("meta", {}).get("machine", {}) or {}
    return (cm.get("cpu") == bm.get("cpu")
            and cm.get("nproc") == bm.get("nproc"))

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
    # --- fuzz (sondes aléatoires pert-vs-GMP — même juge que quality) ---
    f = card.get("fuzz", {})
    for name in f.get("fail_cases", []):
        c = f.get("cases", {}).get(name, {})
        sev = 2 if c.get("verdict") == "ERROR" else 1
        gaps.append(_gap("fuzz", name, "verdict", c.get("verdict", "FAIL"),
                         None, sev,
                         f"fuzz pert≠GMP (seed {f.get('seed')} ; scène : "
                         f"type {c.get('type')} zoom {c.get('zoom')})"))
    for name in f.get("warn_cases", []):
        gaps.append(_gap("fuzz", name, "verdict", "WARN", None, 4,
                         f"fuzz WARN — divergence éparse (seed {f.get('seed')})"))
    # --- precision (juge orbite ×4 : précision d'orbite insuffisante) ---
    pr = card.get("precision", {})
    for name, c in (pr.get("cases") or {}).items():
        if c.get("status") != "ok" or c.get("verdict") == "PASS":
            continue
        sev = 1 if c["verdict"] == "FAIL" else 4
        gaps.append(_gap("precision", name, "diff_px", c.get("diff_px"), None,
                         sev,
                         f"pixels ≠ juge précision ×{pr.get('judge_factor')} "
                         f"({100 * c.get('frac', 0):.2f} %, prec {c.get('prec')} b)"))
    # --- parity ---
    adjudications = load_adjudications()
    p = card.get("parity", {})
    for name, c in p.get("cases", {}).items():
        if c.get("status") == "fractall_fail":
            gaps.append(_gap("parity", name, "status", "fractall_fail", None,
                             1, "rendu fractall échoué"))
        rel = c.get("rel_dsi_pct")
        if rel is not None and rel > 2.0 and c.get("status") == "ok":
            adj = adjudications.get(name)
            note = "divergence parité >2%"
            sev = 2
            if adj:
                note += (f" — ADJUGÉ {adj.get('verdict')} vs GMP "
                         f"({adj.get('date', '?')[:10]}, harness adjudicate)")
                # F3 fautif vs ground truth → pas un gap moteur fractall
                if adj.get("verdict") == "f3_wrong":
                    sev = 4
            else:
                note += " — non adjugé (harness adjudicate " + name + ")"
            gaps.append(_gap("parity", name, "rel_dsi_pct", round(rel, 4),
                             None, sev, note))
    # --- robustesse : cas tués mémoire / en quarantaine (sévérité 2) ---
    s = card.get("speed", {})
    for name, c in s.get("cases", {}).items():
        st = c.get("status", "")
        if st == "quarantined":
            gaps.append(_gap("speed", name, "status", "quarantined", None, 2,
                             "cas en QUARANTAINE — crash/OOM connu, voir "
                             "harness/crash-journal.jsonl"))
        elif st in ("fractall_killed_oom", "fractall_aborted",
                    "fractall_killed"):
            gaps.append(_gap("speed", name, "status", st, None, 2,
                             "cas tué (mémoire) sous cap — voir "
                             "harness/crash-journal.jsonl"))
    # --- speed (triées par ratio décroissant) ---
    speed_gaps = []
    for name, c in s.get("cases", {}).items():
        r = c.get("ratio")
        if c.get("f3_degenerate"):
            # F3 fast-path FAUX sur ce cas (registre f3-degenerate.json) : le
            # ratio n'est pas un gap moteur — listé à part dans le scorecard.
            continue
        if r is not None and r > 2.0:
            speed_gaps.append(_gap("speed", name, "ratio", round(r, 4), None,
                                   3, f"{r:.2f}× plus lent que F3"))
    speed_gaps.sort(key=lambda x: -x["value"])
    gaps += speed_gaps
    gaps += _profile_gaps(card)
    # --- régressions vs baseline (>10%) ---
    if base and _comparable(card, base):
        gaps += _regression_gaps(card, base)
    return gaps

# Seuils de régression vitesse INTRINSÈQUE (même machine, même tier). 15 % et
# un plancher absolu : sous 50 ms le chrono d'un process (démarrage, PNG) n'est
# pas fiable (cf. wisdom-opt `noisy`). Le `spread` mesuré (runs ≥ 2) relève le
# seuil : un delta inférieur au bruit observé n'est pas une régression.
SELF_SPEED_REGRESSION = 1.15
SELF_SPEED_FLOOR_S = 0.050
SELF_GEOMEAN_REGRESSION = 1.10
NS_ITER_REGRESSION = 1.15
BENCH_REGRESSION = 0.85     # iters/s : plus bas = plus lent


def self_speed_ratio(card: dict, base: dict) -> dict:
    """Vitesse fractall vs baseline SANS F3 : ratio médian par cas
    (cur/base, <1 = plus rapide) sur les cas communs mesurés, geomean + pire.
    Même machine ET même tier requis (sinon {}), cf. `_same_machine`."""
    if not (base and _comparable(card, base) and _same_machine(card, base)):
        return {}
    cc = card.get("speed", {}).get("cases", {}) or {}
    bc = base.get("speed", {}).get("cases", {}) or {}
    ratios: dict[str, float] = {}
    for name, c in cc.items():
        cur, prev = c.get("fractall_s"), (bc.get(name) or {}).get("fractall_s")
        if cur and prev and prev > 0:
            ratios[name] = round(cur / prev, 4)
    if not ratios:
        return {}
    geo = math.exp(sum(math.log(r) for r in ratios.values()) / len(ratios))
    worst = max(ratios, key=lambda n: ratios[n])
    return {"geomean": round(geo, 4), "n": len(ratios),
            "worst": {"case": worst, "ratio": ratios[worst]},
            "ratios": ratios, "baseline_sha": base["meta"].get("git_sha")}


def _profile_gaps(card: dict) -> list[dict]:
    """Signaux du profil INDÉPENDANTS de la baseline : une sauvegarde qui domine
    le rendu est une classe de bug perf prouvée (PNG niveau 6, 2026-07-16)."""
    out = []
    for name, c in (card.get("profile", {}).get("cases", {}) or {}).items():
        if c.get("status") != "ok":
            continue
        save, render = c.get("save_s"), c.get("render_s")
        if (save is not None and render is not None and save > render
                and (c.get("wall_s") or 0) > 0.1):
            out.append(_gap("profile", name, "save_s", round(save, 3), None, 3,
                            f"la SAUVEGARDE ({save:.2f}s) domine le rendu "
                            f"({render:.2f}s) — encodeur/colorisation à "
                            f"profiler"))
    return out


def _regression_gaps(card: dict, base: dict) -> list[dict]:
    out = []
    if _same_machine(card, base):
        # --- vitesse intrinsèque par cas (sans F3) ---
        ss = self_speed_ratio(card, base)
        if ss and ss["geomean"] > SELF_GEOMEAN_REGRESSION:
            out.append(_gap("speed", "<self-geomean>", "self_geomean",
                            ss["geomean"], 1.0, 3,
                            f"RÉGRESSION vitesse fractall vs baseline "
                            f"{ss['baseline_sha']} (même machine, >10 %)"))
        cc = card.get("speed", {}).get("cases", {}) or {}
        bc = base.get("speed", {}).get("cases", {}) or {}
        for name, r in sorted((ss.get("ratios") or {}).items(),
                              key=lambda kv: -kv[1]):
            cur = cc[name]["fractall_s"]
            prev = bc[name]["fractall_s"]
            noise = max(cc[name].get("fractall_s_spread") or 0.0,
                        bc[name].get("fractall_s_spread") or 0.0)
            thr = max(SELF_SPEED_REGRESSION, 1.0 + noise)
            if r > thr and cur - prev > SELF_SPEED_FLOOR_S:
                out.append(_gap("speed", name, "fractall_s", cur, prev, 3,
                                f"RÉGRESSION {r:.2f}× vs baseline (même "
                                f"machine ; seuil {thr:.2f})"))
        # --- ns/iter par cas (profil) ---
        cp = card.get("profile", {}).get("cases", {}) or {}
        bp = base.get("profile", {}).get("cases", {}) or {}
        for name, c in cp.items():
            cur, prev = c.get("ns_per_iter"), (bp.get(name) or {}).get("ns_per_iter")
            if cur and prev and cur > prev * NS_ITER_REGRESSION \
                    and c.get("path") == bp[name].get("path"):
                out.append(_gap("profile", name, "ns_per_iter", cur, prev, 3,
                                f"RÉGRESSION boucle pixel {cur/prev:.2f}× "
                                f"(même path {c.get('path')})"))
        # --- débits bench par technique ---
        ce = card.get("bench", {}).get("entries", {}) or {}
        be = base.get("bench", {}).get("entries", {}) or {}
        for key, e in ce.items():
            cur, prev = e.get("iters_per_sec"), (be.get(key) or {}).get("iters_per_sec")
            if cur and prev and cur < prev * BENCH_REGRESSION:
                out.append(_gap("bench", key, "iters_per_sec", cur, prev, 3,
                                f"RÉGRESSION débit {cur/prev:.2f}× vs baseline"))
    # Régression VITESSE : machine-sensible → seulement si même machine que la
    # baseline (HARNESS.md « baseline par machine »). Évite la fausse alarme
    # geomean cross-machine (cf. `_same_machine`).
    cg = card.get("speed", {}).get("geomean_ratio")
    bg = base.get("speed", {}).get("geomean_ratio")
    if cg and bg and cg > bg * 1.10 and _same_machine(card, base):
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
