"""Leviers : où gagner, quand aucun gap ne le dit.

Un gap est un DÉFAUT (pixel faux, régression, regret). Un score sans gap ne
dit pas que le moteur est optimal : il dit que rien n'a régressé. Les leviers
classent le temps MESURÉ par ce qui le consomme, tous axes confondus, pour que
la boucle d'amélioration ait toujours une cible chiffrée :

- profil : secondes par (cas, phase) — orbite, table BLA, pixels, correction
  de fiabilité (`post`), reste ;
- wisdom : coût de la sonde d'itérations, correction de fiabilité des scènes
  en itérations automatiques, regret de routage ;
- truth (volet path standard) : temps de production des scènes jugées.

Les secondes viennent de scènes de tailles différentes : c'est un CLASSEMENT
de pistes, pas un budget. Chaque levier porte la commande qui l'ouvre.
"""

from __future__ import annotations

MIN_SECS = 0.05      # sous ce temps, rien à gagner de mesurable
MIN_SHARE = 0.10     # une phase < 10 % de son cas n'est pas un levier


def _lever(secs, source, case, what, hint):
    return {"secs": round(secs, 3), "source": source, "case": case,
            "what": what, "hint": hint}


def _profile_levers(card: dict) -> list[dict]:
    out = []
    for name, c in (card.get("profile", {}).get("cases") or {}).items():
        if c.get("status") != "ok":
            continue
        wall = c.get("wall_s") or 0.0
        path = c.get("path")
        for phase, secs in (c.get("secs") or {}).items():
            if secs is None or secs < MIN_SECS or secs < MIN_SHARE * wall:
                continue
            share = f"{100 * secs / wall:.0f} % du cas" if wall else ""
            if phase == "post":
                fl = c.get("reliability_flagged_pct")
                what = (f"correction de fiabilité ({fl} % de pixels signalés, "
                        f"{share})" if fl else f"post-traitement ({share})")
                hint = (f"`harness.py ab --cases {name} --env "
                        f"FRACTALL_RELIABILITY=off,escalate` — TODO G14.1")
            elif phase == "orbit":
                what = f"orbite de référence GMP ({share})"
                hint = "séquentielle — TODO G14.3 (`bench_orbit_join`)"
            elif phase == "pixels":
                ns = c.get("ns_per_iter_thread")
                what = (f"boucle pixel `{path}` ({share}"
                        + (f", {ns} ns/iter·thread" if ns else "") + ")")
                hint = f"`harness.py hotspots --case {name}`"
            elif phase == "bla":
                what = f"construction de la table BLA ({share})"
                hint = f"`harness.py hotspots --case {name}`"
            elif phase == "save":
                what = f"sauvegarde PNG ({share})"
                hint = "encodeur / colorisation"
            else:
                what = f"hors phases mesurées ({share})"
                hint = "démarrage, colorisation, allocation"
            out.append(_lever(secs, "profile", name, what, hint))
    return out


def _wisdom_levers(card: dict) -> list[dict]:
    out = []
    for name, c in (card.get("wisdom", {}).get("cases") or {}).items():
        if c.get("status") != "ok":
            continue
        if c.get("kind") == "decision":
            rs = c.get("regret_s")
            if c.get("verdict") in ("WARN", "FAIL") and rs and rs >= MIN_SECS:
                out.append(_lever(
                    rs, "wisdom", name,
                    f"regret de routage ×{c.get('regret')} vs "
                    f"`{c.get('best_equiv')}` à sortie équivalente",
                    "recalibrer le seuil dans `fractal/wisdom.rs`"))
            continue
        render = c.get("render_s") or 0.0
        fs, fp = c.get("fiab_s"), c.get("fiab_pct")
        if fs and fs >= MIN_SECS and fs >= MIN_SHARE * render:
            out.append(_lever(
                fs, "wisdom", name,
                f"correction de fiabilité du path `{c.get('path')}` à "
                f"N auto = {c.get('auto_iterations')} ({fp} % de pixels "
                f"signalés, {100 * fs / render:.0f} % du rendu)",
                "prix de la correction par pixel — TODO G14.1"))
        ps = c.get("probe_s")
        if ps and ps >= MIN_SECS and render and ps >= MIN_SHARE * render:
            out.append(_lever(
                ps, "wisdom", name,
                f"sonde d'itérations automatiques "
                f"({100 * ps / render:.0f} % du rendu)",
                "`fractal/wisdom_iter.rs` (tours, taille de sonde)"))
        wt, tn, tc = (c.get("waste_time_ratio"), c.get("time_at_n_s"),
                      c.get("time_at_sufficient_s"))
        if wt and tn and tc and tn - tc >= MIN_SECS and wt > 1.10:
            out.append(_lever(
                tn - tc, "wisdom", name,
                f"plafond auto sur-provisionné : N/{c.get('waste')} "
                f"suffisait (×{wt} en temps)",
                "`starting_iterations` / règle de réduction"))
    return out


def _truth_levers(card: dict) -> list[dict]:
    out = []
    for name, c in (card.get("truth", {}).get("cases") or {}).items():
        secs = c.get("secs")
        if (c.get("status") == "ok" and name.startswith("std:") and secs
                and secs >= 4 * MIN_SECS):
            out.append(_lever(secs, "truth", name,
                              f"rendu de production `{c.get('path')}` "
                              f"(256×160)",
                              "`FRACTALL_PERTURB_STATS=1` : part du traqueur "
                              "et de la correction"))
    return out


def compute_levers(card: dict) -> list[dict]:
    lv = _profile_levers(card) + _wisdom_levers(card) + _truth_levers(card)
    lv.sort(key=lambda x: -x["secs"])
    return lv


def levers_md(levers: list[dict], top: int = 8) -> list[str]:
    L = ["## Leviers (où gagner — temps mesuré, trié)", ""]
    if not levers:
        L += ["_aucun levier au-dessus du plancher de mesure "
              "(axes profile / wisdom absents ?)_", ""]
        return L
    L += ["_Classement de pistes, pas un budget : les scènes n'ont pas toutes "
          "la même taille. Un levier n'est pas un défaut — c'est là que le "
          "temps va._", "",
          "| # | s | Source | Cas | Ce qui coûte | Pour l'ouvrir |",
          "|---:|---:|---|---|---|---|"]
    for i, x in enumerate(levers[:top], 1):
        L.append(f"| {i} | {x['secs']:.2f} | {x['source']} | `{x['case']}` | "
                 f"{x['what']} | {x['hint']} |")
    if len(levers) > top:
        L.append(f"\n_… {len(levers) - top} autre(s) dans le history JSON "
                 f"(`harness.py levers`)._")
    L.append("")
    return L


def print_levers(levers: list[dict], top: int = 8) -> None:
    if not levers:
        return
    print(f"\nLeviers — où va le temps mesuré (top {min(top, len(levers))}) :")
    for i, x in enumerate(levers[:top], 1):
        print(f"  {i:>2}. {x['secs']:>6.2f} s  {x['source']}/{x['case']} — "
              f"{x['what']}\n        → {x['hint']}")
