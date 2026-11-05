"""Point d'entrée argparse de scripts/harness.py."""

from __future__ import annotations

import argparse

from .commands import (
    cmd_render,
    cmd_adjudicate, cmd_baseline, cmd_gaps, cmd_journal, cmd_score,
)
from .ab import cmd_ab
from .commands_insight import cmd_doctor, cmd_trend
from .commands_guard import cmd_preflight, cmd_quarantine, cmd_slow_safe
from .config import FUZZ_SEED_DEFAULT
from .guard import install_signal_handlers
from .wisdom_opt import cmd_wisdom_opt

USAGE = """Scorecard orchestrator — mesure fractall vs Fraktaler-3 (protocole HARNESS.md).

Produit `harness/history/<UTC>-<sha>.json` + `SCORECARD.md` (gaps triés).
Axes : speed (ratio wall-clock vs F3), parity (compare_f3.py), quality
(fractall-quality suite vs GMP), fuzz (sondes pert vs GMP), goldens (cargo test
golden_images).

F3 auto-détecté (F3_BIN sinon fraktaler-3-3.1/) ; sans lui speed/parity sont
`f3_unavailable` et le reste tourne. Sans numpy/OpenEXR : `f3_deps_missing`.
"""


ALL_AXES = "speed,parity,quality,fuzz,precision,goldens,profile,bench"
DEFAULT_AXES = ALL_AXES


def main() -> None:
    install_signal_handlers()  # Ctrl-C/SIGTERM → pas de fausse quarantaine
    ap = argparse.ArgumentParser(
        description=USAGE,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    sc = sub.add_parser("score", help="Mesure et écrit un scorecard.")
    sc.add_argument("--tier", choices=["quick", "standard", "full"],
                    default="quick")
    sc.add_argument("--axes", default=DEFAULT_AXES,
                    help=f"CSV parmi {ALL_AXES} (défaut : tous)")
    sc.add_argument("--fuzz-seed", type=int, default=FUZZ_SEED_DEFAULT,
                    help="seed des sondes fuzz (défaut committé ; le faire "
                         "tourner au rebaseline)")
    sc.add_argument("--parity-iterations", type=int, default=0,
                    help="plafond d'itérations COMMUN fractall/F3 pour l'axe "
                         "parity (0 = natif ; utile en tier full)")
    sc.add_argument("--no-rebuild", action="store_true")
    sc.add_argument("--cases", default=None,
                    help="liste CSV de stems (override du tier)")
    sc.add_argument("--width", type=int, default=None)
    sc.add_argument("--height", type=int, default=None)
    sc.add_argument("--runs", type=int, default=None)
    sc.add_argument("--timeout", type=float, default=None)
    sc.add_argument("--mem-limit-mb", type=int, default=None,
                    help="cap RLIMIT_AS par process (défaut 85%% RAM ; "
                         "0 = off)")
    sc.add_argument("--no-mem-limit", action="store_true",
                    help="désactive le cap mémoire (⚠️ risque plantage OS)")
    sc.add_argument("--no-quarantine", action="store_true",
                    help="ne pas skipper les cas en quarantaine")
    sc.set_defaults(func=cmd_score)

    pf = sub.add_parser("preflight",
                        help="Vet le corpus sous cap mémoire (trouve le cas "
                             "qui fait planter l'OS AVANT le sweep).")
    pf.add_argument("--tier", choices=["quick", "standard", "full"],
                    default=None, help="cases du tier (défaut : tout le corpus)")
    pf.add_argument("--cases", default=None, help="liste CSV de stems")
    pf.add_argument("--width", type=int, default=None)
    pf.add_argument("--height", type=int, default=None)
    pf.add_argument("--timeout", type=float, default=None)
    pf.add_argument("--mem-limit-mb", type=int, default=None)
    pf.add_argument("--no-mem-limit", action="store_true")
    pf.add_argument("--no-rebuild", action="store_true")
    pf.add_argument("--flag-rss-mb", type=int, default=None,
                    help="seuil pic RSS (MB) au-delà duquel un cas est "
                         "quarantainé (défaut 60%% du cap)")
    pf.add_argument("--no-quarantine", action="store_true",
                    help="mesurer sans quarantainer les gourmands")
    pf.set_defaults(func=cmd_preflight)

    qs = sub.add_parser("quarantine",
                        help="Gère les cas exclus des sweeps (crash/OOM).")
    qs.add_argument("action", choices=["list", "add", "remove", "clear"])
    qs.add_argument("case", nargs="?", default=None)
    qs.add_argument("--reason", default=None)
    qs.set_defaults(func=cmd_quarantine)

    ss = sub.add_parser("slow-safe",
                        help="Gère les cas corrects+memory-safe hors budget "
                             "temps (skip sweeps, PAS un gap crash/OOM).")
    ss.add_argument("action", choices=["list", "add", "remove"])
    ss.add_argument("case", nargs="?", default=None)
    ss.add_argument("--note", default=None)
    ss.set_defaults(func=cmd_slow_safe)

    wo = sub.add_parser("wisdom-optimality",
                        help="Vérifie que le plan choisi par le wisdom n'est "
                             "jamais battu >10%% par un plan forcé (critère G9).")
    wo.add_argument("--cases", default=None,
                    help="CSV de stems (défaut : échantillon 2-régimes)")
    wo.add_argument("--size", type=int, default=None)
    wo.add_argument("--runs", type=int, default=None)
    wo.add_argument("--no-rebuild", action="store_true")
    wo.set_defaults(func=cmd_wisdom_opt)

    js = sub.add_parser("journal",
                        help="Affiche le journal des incidents (crashes/OOM).")
    js.set_defaults(func=cmd_journal)

    rd = sub.add_parser("render",
                        help="Régénère SCORECARD.md depuis un history (sans "
                             "re-mesurer ; gaps recalculés vs baseline).")
    rd.add_argument("history", nargs="?", default=None,
                    help="fichier harness/history/*.json (défaut : le dernier)")
    rd.add_argument("--update-history", action="store_true",
                    help="réécrire aussi les gaps dans le history")
    rd.set_defaults(func=cmd_render)

    bl = sub.add_parser("baseline",
                        help="Fige le dernier history de CETTE machine comme "
                             "sa baseline (harness/baselines/<machine>.json).")
    bl.add_argument("history", nargs="?", default=None,
                    help="history à figer (défaut : dernier de cette machine)")
    bl.add_argument("--legacy", action="store_true",
                    help="écrire le repli historique harness/baseline.json")
    bl.set_defaults(func=cmd_baseline)

    aj = sub.add_parser("adjudicate",
                        help="Arbitre 3-voies fractall/F3/GMP d'un cas parité "
                             "et persiste le verdict (fin du re-litige).")
    aj.add_argument("cases", nargs="+", help="stems toml/ à adjuger")
    aj.add_argument("--size", type=int, default=96,
                    help="résolution (GMP per-pixel ~1µs/iter — petit sur "
                         "les cas profonds)")
    aj.set_defaults(func=cmd_adjudicate)

    gp = sub.add_parser("gaps", help="Ré-affiche les gaps du dernier history.")
    gp.set_defaults(func=cmd_gaps)

    ab = sub.add_parser("ab",
                        help="A/B isolé : variantes d'env (--env K=a,b) ou de "
                             "flags CLI (--flags '--x|'), médiane + bruit + "
                             "phases + identité de sortie (méthode G14).")
    ab.add_argument("--cases", required=True, help="CSV de stems toml/")
    ab.add_argument("--env", action="append", default=[],
                    help="K=v1,v2,… (répétable ; vide = variable posée vide)")
    ab.add_argument("--flags", action="append", default=[],
                    help="alternatives séparées par | (ex. '--dd-tier|')")
    ab.add_argument("--size", type=int, default=None)
    ab.add_argument("--runs", type=int, default=None)
    ab.add_argument("--timeout", type=float, default=None)
    ab.add_argument("--no-rebuild", action="store_true")
    ab.set_defaults(func=cmd_ab)

    dr = sub.add_parser("doctor",
                        help="Prérequis machine, axes mesurables ici, état de "
                             "la baseline (à lancer en début de session).")
    dr.add_argument("--strict", action="store_true",
                    help="code retour 1 si un prérequis BLOQUANT manque")
    dr.set_defaults(func=cmd_doctor)

    tr = sub.add_parser("trend",
                        help="Évolution des métriques clés dans harness/history "
                             "(même machine par défaut).")
    tr.add_argument("--last", type=int, default=12, help="N derniers scores")
    tr.add_argument("--tier", choices=["quick", "standard", "full"], default=None)
    tr.add_argument("--all-machines", action="store_true")
    tr.add_argument("--case", default=None,
                    help="stem : temps/ratio/ns-iter de CE cas au fil des scores")
    tr.add_argument("--bench", action="store_true",
                    help="débits --wisdom-bench par technique")
    tr.add_argument("--out", default=None, help="écrire le markdown ici")
    tr.set_defaults(func=cmd_trend)

    args = ap.parse_args()
    args.func(args)
