"""Axe wisdom-optimality (G9) : le plan choisi par le wisdom n'est jamais battu >10 %."""

from __future__ import annotations

import json
import os
import re
import subprocess

from .config import (
    BENCH, CLI, REPO, TOML_DIR, _now_iso, cargo_build, git_sha, machine_info,
)
from .gaps import load_wisdom_adjudications
from .guard import timed_runs

# Échantillon par défaut de l'axe wisdom-optimality : cas Mandelbrot f64-tier
# couvrant les DEUX régimes de routage harmonic — période courte (LLA gagne,
# auto doit router harmonic) ET période longue (BLA gagne, auto doit router
# BLA). Teste le seuil `route_harmonic_auto` (period0≤100) des deux côtés.
WISDOM_OPT_SAMPLE = [
    "flake", "test3", "mitosis", "glitch_test_5",   # courtes → harmonic gagne
    "dinosaur_fossils",                             # p78 borderline (gagne peu)
    "e50", "e113", "dragon",                        # longues → BLA gagne
]

# Variantes de plan forçables via env sur la dimension ROUTAGE que le wisdom
# décide activement (harmonic vs BLA, G9.3). `auto` = le plan CHOISI par le
# wisdom ; les autres = alternatives forcées à départager. (Les dimensions
# tier/device sont couvertes par 9.6/9.5 ; ici on verrouille le routage.)
WISDOM_OPT_VARIANTS = {
    "auto": {"FRACTALL_HARMONIC_LA": "auto"},   # décision du wisdom
    "bla":  {"FRACTALL_HARMONIC_LA": "bla"},    # BLA forcée
    "lla":  {"FRACTALL_HARMONIC_LA": "lla"},    # Harmonic LLA forcée
}

def _wisdom_opt_path(name: str, width: int, height: int, env: dict) -> str:
    """Rend une fois et extrait le `[FRACTALL] … path=…` (technique routée)."""
    toml_path = TOML_DIR / f"{name}.toml"
    out = BENCH / "wisdom-opt"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [str(CLI), "--toml", str(toml_path), "--width", str(width),
           "--height", str(height), "--bailout", "25",
           "--output", str(out / f"{name}_probe.png")]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           env=env, text=True, timeout=600)
    except Exception:
        return "?"
    m = re.findall(r"path=(\S+)", p.stdout or "")
    return m[-1] if m else "?"

def cmd_wisdom_opt(args) -> None:
    """Axe wisdom-optimality (critère d'excellence G9) : pour un échantillon,
    chronométrer le plan CHOISI par le wisdom (`auto`) contre les plans forcés
    alternatifs sur la dimension de routage harmonic, et vérifier que le choix
    n'est JAMAIS battu > 10 % (vitesse) par une alternative viable. Un FAIL =
    le seuil de routage (`route_harmonic_auto`) est mal calibré sur ce cas.

    Toutes les variantes sont le MÊME tier numérique (perturbation f64) → même
    correction par construction ; l'axe mesure donc la vitesse. La correction
    inter-tier (dd/exp escalade) relève de 9.6. CPU-only, sans effet render."""
    cases = ([c.strip() for c in args.cases.split(",")] if args.cases
             else list(WISDOM_OPT_SAMPLE))
    width = args.size or 256
    height = args.size or 256
    runs = args.runs or 3
    warn_r, fail_r = 1.10, 1.25
    wisdom_adj = load_wisdom_adjudications()  # tradeoffs correction déjà tranchés
    base_env = os.environ.copy()
    base_env["FRACTALL_NO_AUTO_ADJUST"] = "1"
    base_env["FRACTALL_NO_PERIOD"] = "1"   # aligné prod + axe vitesse (Brent OFF)

    if not args.no_rebuild and not CLI.exists():
        cargo_build("fractall-cli")
    (BENCH / "wisdom-opt").mkdir(parents=True, exist_ok=True)  # sinon --output échoue
    print(f"→ wisdom-optimality {len(cases)} cas · {width}×{height} · "
          f"runs={runs} · seuil PASS≤{warn_r:.2f} FAIL>{fail_r:.2f}")

    rows: list[dict] = []
    n_pass = n_warn = n_fail = n_adj = 0
    for i, name in enumerate(cases, 1):
        toml_path = TOML_DIR / f"{name}.toml"
        print(f"  [{i:>2}/{len(cases)}] {name:20s}", end=" ", flush=True)
        if not toml_path.exists():
            print("✗ toml absent")
            continue
        times: dict[str, float | None] = {}
        for var, ov in WISDOM_OPT_VARIANTS.items():
            env = dict(base_env)
            env.update(ov)
            cmd = [str(CLI), "--toml", str(toml_path), "--width", str(width),
                   "--height", str(height), "--bailout", "25",
                   "--output", str(BENCH / "wisdom-opt" / f"{name}_{var}.png")]
            st, med = timed_runs(cmd, env, 600.0, runs)
            times[var] = med if st == "ok" else None
        chosen = times.get("auto")
        alts = {v: t for v, t in times.items() if v != "auto" and t is not None}
        if chosen is None or not alts:
            print("⊘ mesure incomplète")
            rows.append({"case": name, "status": "incomplete", "times": times})
            continue
        best_var = min(alts, key=lambda v: alts[v])
        best_alt = alts[best_var]
        ratio = chosen / best_alt if best_alt > 0 else 1.0
        routed = _wisdom_opt_path(name, width, height,
                                  {**base_env, "FRACTALL_HARMONIC_LA": "auto"})
        # Sortie de l'alternative rapide IDENTIQUE à celle du wisdom ? (encodeur
        # PNG déterministe → mêmes pixels ⇔ mêmes octets). Différente = le plan
        # rapide N'est PAS le même rendu → une avance de vitesse peut être un
        # tradeoff correction que le wisdom fait EXPRÈS (router le plus lent mais
        # correct). Un vrai gap de vitesse = plus lent À SORTIE IDENTIQUE.
        odir = BENCH / "wisdom-opt"
        differs = None
        try:
            differs = ((odir / f"{name}_auto.png").read_bytes()
                       != (odir / f"{name}_{best_var}.png").read_bytes())
        except Exception:
            pass
        # Plancher de bruit : sous 60 ms d'écart absolu, le chrono (rendu <~200 ms,
        # machine possiblement chargée) n'est pas fiable — un ratio élevé y est du
        # bruit, pas une sous-optimalité du wisdom. On force PASS (noté `noisy`).
        noisy = abs(chosen - best_alt) < 0.060
        if noisy or ratio <= warn_r:
            verdict = "PASS"
        elif differs:
            # plus lent mais l'alternative rapide rend AUTRE CHOSE → adjudication
            # correction requise avant de conclure ; pas un gap de vitesse net.
            verdict = "ADJUDICATE"
        else:
            verdict = "WARN" if ratio <= fail_r else "FAIL"
        # Adjudication persistée : un ADJUDICATE connu (wisdom route le path
        # lent-mais-CORRECT, l'alternative rapide est fausse vs GMP) est marqué
        # « déjà-adjugé✓ » tant que le RÉGIME est stable — pas de re-litige. Le
        # verdict brut reste ADJUDICATE (aucune sémantique de comptage touchée) ;
        # un changement de routed_path OU la disparition de output_differs =
        # `⚠ RÉGIME CHANGÉ` (l'adjudication ne tient peut-être plus → ré-adjuger).
        adj = wisdom_adj.get(name) if verdict == "ADJUDICATE" else None
        adj_status = None
        if adj is not None:
            regime_ok = (adj.get("routed_path") == routed and differs is True)
            adj_status = "known" if regime_ok else "regime_changed"
        n_pass += verdict == "PASS"
        n_warn += verdict == "WARN"
        n_fail += verdict == "FAIL"
        n_adj += verdict == "ADJUDICATE"
        rows.append({"case": name, "status": "ok", "verdict": verdict,
                     "ratio_chosen_vs_best": round(ratio, 3), "noisy": noisy,
                     "output_differs": differs, "adj_status": adj_status,
                     "chosen_s": round(chosen, 4), "best_alt": best_var,
                     "best_alt_s": round(best_alt, 4),
                     "routed_path": routed, "times": times})
        tag = " noisy" if noisy else " sortie≠" if differs else ""
        if adj_status == "known":
            tag = " déjà-adjugé✓"
        elif adj_status == "regime_changed":
            tag = " ⚠ RÉGIME CHANGÉ"
        print(f"[{verdict}{tag}] auto={chosen*1000:.0f}ms routed={routed} "
              f"vs best({best_var})={best_alt*1000:.0f}ms ratio={ratio:.2f}")

    out = BENCH / "wisdom-opt"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"date_utc": _now_iso(), "git_sha": git_sha(),
               "machine": machine_info(), "size": width, "runs": runs,
               "warn_ratio": warn_r, "fail_ratio": fail_r,
               "n_pass": n_pass, "n_warn": n_warn, "n_fail": n_fail,
               "n_adjudicate": n_adj, "cases": rows}
    (out / "wisdom-opt.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    L = [f"# Wisdom-optimality — {_now_iso()} (`{git_sha()}`)", "",
         f"- {width}² · runs={runs} · PASS≤{warn_r} WARN≤{fail_r} FAIL>{fail_r}",
         "- `auto` = plan choisi par le wisdom ; comparé au meilleur plan forcé "
         "(bla/lla). ratio>1 = le wisdom est plus lent que l'alternative.",
         "- **FAIL** = plus lent À SORTIE IDENTIQUE (vrai gap). **ADJUDICATE** = "
         "plus lent mais l'alternative rend AUTRE CHOSE (tradeoff correction "
         "possiblement voulu — vérifier vs GMP avant de recalibrer).", "",
         f"**{n_pass} PASS · {n_warn} WARN · {n_fail} FAIL · "
         f"{n_adj} ADJUDICATE**", "",
         "| Cas | Verdict | routé | ratio auto/best | auto (ms) | best alt | "
         "sortie≠ |",
         "|---|---|---|---:|---:|---|---|"]
    for r in rows:
        if r["status"] != "ok":
            L.append(f"| {r['case']} | {r['status']} | — | — | — | — | — |")
            continue
        d = {True: "oui", False: "non", None: "?"}[r.get("output_differs")]
        vcell = r["verdict"]
        if r.get("adj_status") == "known":
            vcell += " (déjà-adjugé✓)"
        elif r.get("adj_status") == "regime_changed":
            vcell += " (⚠ RÉGIME CHANGÉ)"
        L.append(f"| {r['case']} | {vcell} | {r['routed_path']} | "
                 f"{r['ratio_chosen_vs_best']} | {r['chosen_s']*1000:.0f} | "
                 f"{r['best_alt']} {r['best_alt_s']*1000:.0f}ms | {d} |")
    (out / "wisdom-opt.md").write_text("\n".join(L) + "\n")
    print(f"\n✓ rapport : {(out / 'wisdom-opt.md').relative_to(REPO)}")
    print(f"→ **{n_pass} PASS · {n_warn} WARN · {n_fail} FAIL · "
          f"{n_adj} ADJUDICATE**")
    n_adj_known = sum(r.get("adj_status") == "known" for r in rows)
    n_adj_changed = sum(r.get("adj_status") == "regime_changed" for r in rows)
    n_adj_fresh = n_adj - n_adj_known - n_adj_changed
    if n_fail:
        print("⚠ wisdom SUB-OPTIMAL (plus lent à sortie IDENTIQUE) — seuil "
              "routage à recalibrer, voir le rapport.")
    if n_adj_fresh:
        print(f"• ADJUDICATE (×{n_adj_fresh} NOUVEAU) : alternative plus rapide "
              "mais sortie ≠ — vérifier correction vs GMP, puis persister dans "
              "harness/wisdom-adjudications.json.")
    if n_adj_known:
        print(f"• ADJUDICATE (×{n_adj_known} déjà-adjugé✓) : tradeoff correction "
              "connu (wisdom route le lent-mais-correct) — rien à faire.")
    if n_adj_changed:
        print(f"⚠ ADJUDICATE (×{n_adj_changed} RÉGIME CHANGÉ) : le routage/sortie "
              "diffère de l'adjudication persistée — RÉ-ADJUGER vs GMP.")
