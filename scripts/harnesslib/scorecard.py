"""Rendu SCORECARD.md + écriture/lecture de harness/history."""

from __future__ import annotations

from pathlib import Path
import json

from .config import HISTORY_DIR, MARKER, SEVERITY_LABEL
from .gaps import _comparable, _same_machine, sort_gaps

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
    if base:
        bmeta = base.get("meta", {})
        L.append(f"- **Baseline** : `{base.get('_file') or m.get('baseline_file') or 'harness/baseline.json'}` "
                 f"({bmeta.get('git_sha')} · tier {bmeta.get('tier')} · "
                 f"{bmeta.get('date_utc', '?')[:10]})"
                 + ("" if _same_machine(card, base) else " · **autre machine**"))
    if base and not comparable:
        L.append(f"- _baseline présente mais tier différent "
                 f"({base.get('meta', {}).get('tier')}) — pas de delta._")
    L.append("")

    # Speed
    s = card.get("speed")
    if s and s.get("status") != "skipped":
        L += ["## Vitesse (ratio fractall/F3, <1 = fractall gagne)", ""]
        if s.get("status") in ("f3_unavailable", "f3_deps_missing"):
            L.append("_F3 indisponible — timings fractall seuls "
                     "(voir history JSON)._\n")
        # Delta vitesse : seulement si même machine (ratios machine-sensibles,
        # cf. `_same_machine`). Sur une autre machine, on n'affiche pas de delta
        # trompeur (ni fausse alarme geomean côté gaps).
        speed_comparable = bool(b and _same_machine(card, base))
        bs = (b or {}).get("speed", {}) if speed_comparable else {}
        if b and not speed_comparable:
            L.append("_baseline d'une autre machine — pas de delta vitesse "
                     "(ratios machine-sensibles ; correction comparée ci-dessous)._\n")
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
        excl = s.get("excluded_f3_degenerate") or []
        if excl:
            names = ", ".join(
                f"{n} ({_fmt(s['cases'][n].get('ratio'))})" for n in excl)
            L.append(f"| exclus (F3-dégénéré, ratio sans objet) | {names} | |")
        slow = s.get("excluded_slow_safe") or []
        if slow:
            L.append(f"| exclus (slow-safe, correct hors budget temps) | "
                     f"{', '.join(slow)} | |")
        L.append("")
        # Vitesse INTRINSÈQUE (sans F3) : temps fractall par cas, comparés à
        # la baseline même machine/tier (régression détectable sans F3).
        ss = card.get("speed_self") or {}
        L += ["### Temps fractall par cas (vitesse intrinsèque)", ""]
        if ss:
            w = ss.get("worst") or {}
            L.append(f"- **auto-ratio vs baseline `{ss.get('baseline_sha')}`** "
                     f"(même machine, geomean cur/base sur {ss.get('n')} cas, "
                     f"<1 = plus rapide) : **{_fmt(ss.get('geomean'))}** · pire "
                     f"{_fmt(w.get('ratio'))} ({w.get('case')})")
        elif base and not speed_comparable:
            L.append("- _pas d'auto-ratio : baseline d'une autre machine._")
        elif base and not comparable:
            L.append("- _pas d'auto-ratio : baseline d'un autre tier._")
        L.append(f"- temps cumulé : **{_fmt(s.get('fractall_total_s'))} s** "
                 f"sur {s.get('n_measured', 0)} cas mesurés")
        L.append("")
        L.append("| Cas | fractall (s) | min (s) | bruit | vs baseline | ratio F3 |")
        L.append("|---|---:|---:|---:|---:|---:|")
        bcases = (bs.get("cases") or {}) if speed_comparable else {}
        for name, c in s.get("cases", {}).items():
            fs = c.get("fractall_s")
            if fs is None:
                L.append(f"| `{name}` | — ({c.get('status')}) | | | | |")
                continue
            prev = (bcases.get(name) or {}).get("fractall_s")
            vsb = f"{fs / prev:.2f}×" if prev else "—"
            sp = c.get("fractall_s_spread")
            L.append(f"| `{name}` | {fs:.3f} | {_fmt(c.get('fractall_s_min'))} | "
                     f"{(f'{100*sp:.0f} %' if sp is not None else '—')} | {vsb} | "
                     f"{_fmt(c.get('ratio'))} |")
        L.append("")

    # Parity
    p = card.get("parity")
    if p and p.get("status") != "skipped":
        L += ["## Parité (compare_f3 — Δsmooth-iter vs F3)", ""]
        if p.get("status") == "f3_unavailable":
            L.append("_F3 indisponible — axe non exécuté._\n")
        elif p.get("status") == "f3_deps_missing":
            L.append("_F3 présent mais deps Python EXR absentes (numpy/"
                     "OpenEXR/Pillow) — axe non exécuté (`harness.py doctor`)._\n")
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

    # Fuzz
    f = card.get("fuzz")
    if f and f.get("status") != "skipped":
        L += ["## Fuzz (sondes aléatoires pert vs GMP)", ""]
        if f.get("status") != "ok":
            L.append(f"_axe indisponible : {f.get('status')}._\n")
        else:
            bf = (b or {}).get("fuzz", {})
            base_txt = ""
            if bf and bf.get("seed") == f.get("seed"):
                base_txt = (f" (baseline : {bf.get('n_pass')}P/"
                            f"{bf.get('n_warn')}W/{bf.get('n_fail')}F)")
            L.append(f"- seed `{f['seed']}` · {f['n_probes']} sondes → "
                     f"**{f['n_pass']} PASS · {f['n_warn']} WARN · "
                     f"{f['n_fail']} FAIL**{base_txt}")
            for name in (f.get("fail_cases", []) + f.get("warn_cases", [])):
                c = f["cases"][name]
                L.append(f"  - `{name}` **{c['verdict']}** — "
                         f"c=({c['center_x']}, {c['center_y']}) "
                         f"zoom {c['zoom']} iters {c['iterations']}")
            L.append("")

    # Goldens
    g = card.get("goldens")
    if g and g.get("status") != "skipped":
        status = "🟢 VERT" if g.get("passed") else "🔴 ROUGE"
        L += ["## Goldens (pixel-exact)", "", f"- {status}", ""]

    # Profil : où va le temps
    pr = card.get("profile")
    if pr and pr.get("status") != "skipped":
        L += ["## Profil — où va le temps (sans F3, 1 run par cas)", ""]
        if pr.get("status") != "ok":
            L.append(f"_axe indisponible : {pr.get('status')}._\n")
        else:
            cs = pr.get("cumulative_share_pct") or {}
            L.append("- **répartition cumulée** : "
                     + " · ".join(f"{k} {cs.get(k)} %" for k in
                                  ("orbit", "pixels", "post", "save", "other"))
                     + f" (sur {pr.get('n_ok')} cas)")
            dc = pr.get("dominant_counts") or {}
            if dc:
                L.append("- **phase dominante** : "
                         + ", ".join(f"{k} ×{v}" for k, v in
                                     sorted(dc.items(), key=lambda kv: -kv[1])))
            if pr.get("dd_escalations"):
                L.append(f"- **escalade dd** (frame re-rendue au tier dd) : "
                         f"{', '.join(pr['dd_escalations'])}")
            L.append(f"- paths : {', '.join(pr.get('paths') or []) or '—'}")
            L.append("")
            L.append("| Cas | path | variantes | wall (s) | orbite % | pixels % | "
                     "post % | save % | autre % | ns/iter | iter/px | esc.dd | "
                     "fiab. % |")
            L.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|")
            for name, c in pr.get("cases", {}).items():
                if c.get("status") != "ok":
                    L.append(f"| `{name}` | — ({c.get('status')}) | | | | | | | | | | | |")
                    continue
                sh = c.get("share_pct") or {}
                rel = c.get("reliability_flagged_pct")
                L.append(f"| `{name}` | {c.get('path') or '?'} | {c.get('variants') or '—'} | "
                         f"{c.get('wall_s'):.3f} | {sh.get('orbit')} | {sh.get('pixels')} | "
                         f"{sh.get('post')} | {sh.get('save')} | {sh.get('other')} | "
                         f"{_fmt(c.get('ns_per_iter'))} | "
                         f"{_fmt0(c.get('avg_iter_px'))} | "
                         f"{'oui' if c.get('dd_escalation') else ''} | "
                         f"{(f'{rel:.2f}' if rel is not None else '—')} |")
            L.append("")

    # Bench : débits par technique
    be = card.get("bench")
    if be and be.get("status") != "skipped":
        L += ["## Bench — débit par technique (`--wisdom-bench` isolé)", ""]
        if be.get("status") != "ok":
            L.append(f"_axe indisponible : {be.get('status')}._\n")
        else:
            bb = ((b or {}).get("bench", {}).get("entries", {})
                  if (b and _same_machine(card, base)) else {})
            L.append("| Technique | iters/s | ns/iter | taille | vs baseline |")
            L.append("|---|---:|---:|---:|---:|")
            for key, e in be.get("entries", {}).items():
                prev = (bb.get(key) or {}).get("iters_per_sec")
                vsb = f"{e['iters_per_sec'] / prev:.2f}×" if prev else "—"
                L.append(f"| {key} | {e['iters_per_sec']:.3e} | "
                         f"{_fmt(e.get('ns_per_iter'))} | {e.get('width')}² | {vsb} |")
            L.append("")

    # Tendance (même machine, derniers runs)
    hist = load_history(same_machine_as=card, limit=6)
    if len(hist) > 1:
        L += ["## Tendance (même machine, 6 derniers scores)", ""]
        L += trend_table(hist)
        L.append("")
        L.append("_`python3 scripts/harness.py trend` pour l'historique complet "
                 "(`--case <stem>` pour un cas)._")
        L.append("")

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
    L.append("_Scorecards versionnés : `harness/history/` · baseline : "
             "`harness/baseline.json`. Généré par `scripts/harness.py`._")
    return "\n".join(L) + "\n"

def _fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


def _fmt0(v):
    return f"{v:.0f}" if isinstance(v, (int, float)) else "—"

def load_history(same_machine_as: dict | None = None,
                 tier: str | None = None, limit: int | None = None) -> list[dict]:
    """Cartes de `harness/history/` triées par date croissante, filtrées
    (même machine que `same_machine_as`, tier). Fichiers illisibles ignorés."""
    cards = []
    for path in sorted(HISTORY_DIR.glob("*.json")):
        try:
            c = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(c, dict) or "meta" not in c:
            continue
        if same_machine_as and not _same_machine(c, same_machine_as):
            continue
        if tier and c["meta"].get("tier") != tier:
            continue
        c["_file"] = path.name
        cards.append(c)
    return cards[-limit:] if limit else cards


def trend_row(c: dict) -> dict:
    """Ligne de tendance (métriques clés) d'une carte — tolère le schéma 1."""
    m = c.get("meta", {})
    s, p, q = c.get("speed", {}) or {}, c.get("parity", {}) or {}, c.get("quality", {}) or {}
    f, g = c.get("fuzz", {}) or {}, c.get("goldens", {}) or {}
    prof = c.get("profile", {}) or {}
    share = prof.get("cumulative_share_pct") or {}
    return {
        "date": m.get("date_utc", "?")[:16].replace("T", " "),
        "sha": m.get("git_sha", "?"), "tier": m.get("tier", "?"),
        "geomean": s.get("geomean_ratio"),
        "wins": len(s.get("wins", []) or []) if s.get("cases") else None,
        "fractall_total_s": s.get("fractall_total_s"),
        "self_geomean": (c.get("speed_self") or {}).get("geomean"),
        "parity_ok": p.get("n_ok") if p.get("cases") else None,
        "parity_equiv": p.get("n_pixel_equiv") if p.get("cases") else None,
        "q": (q.get("n_pass"), q.get("n_warn"), q.get("n_fail")) if q.get("presets") else None,
        "fuzz": (f.get("n_pass"), f.get("n_warn"), f.get("n_fail")) if f.get("cases") else None,
        "goldens": g.get("passed"),
        "pixels_share": share.get("pixels"), "orbit_share": share.get("orbit"),
        "n_gaps": len(c.get("gaps", []) or []),
    }


def _pwf(v):
    return f"{v[0]}/{v[1]}/{v[2]}" if v else "—"


def trend_table(cards: list[dict]) -> list[str]:
    L = ["| Date (UTC) | Commit | Tier | geomean/F3 | wins | fractall Σs | "
         "auto-ratio | parité ok/équiv | Q P/W/F | fuzz P/W/F | goldens | "
         "px % | gaps |",
         "|---|---|---|---:|---:|---:|---:|---|---|---|---|---:|---:|"]
    for c in cards:
        r = trend_row(c)
        L.append(f"| {r['date']} | `{r['sha']}` | {r['tier']} | "
                 f"{_fmt(r['geomean'])} | {r['wins'] if r['wins'] is not None else '—'} | "
                 f"{_fmt(r['fractall_total_s'])} | {_fmt(r['self_geomean'])} | "
                 f"{r['parity_ok'] if r['parity_ok'] is not None else '—'}/"
                 f"{r['parity_equiv'] if r['parity_equiv'] is not None else '—'} | "
                 f"{_pwf(r['q'])} | {_pwf(r['fuzz'])} | "
                 f"{'🟢' if r['goldens'] else ('🔴' if r['goldens'] is False else '—')} | "
                 f"{r['pixels_share'] if r['pixels_share'] is not None else '—'} | "
                 f"{r['n_gaps']} |")
    return L


def latest_history() -> Path | None:
    files = sorted(HISTORY_DIR.glob("*.json"))
    return files[-1] if files else None

def write_history(card: dict) -> Path:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    m = card["meta"]
    path = HISTORY_DIR / f"{m['stamp']}-{m['git_sha']}.json"
    path.write_text(json.dumps(card, indent=2, ensure_ascii=False))
    return path
