"""Commande `wisdom-optimality` : l'axe `wisdom` seul, échantillon complet.

Enveloppe de `axes_wisdom.axis_wisdom` (l'axe tourne aussi dans `score`).
Sert au cycle ciblé sur une décision (`--only harmonic|algorithm|decision|
iter`) sans payer les autres axes. Rapport : `bench/harness/wisdom/`.
"""

from __future__ import annotations

import json

from .axes_wisdom import axis_wisdom, wisdom_gaps, wisdom_md
from .config import BENCH, CLI, REPO, _now_iso, cargo_build, git_sha, machine_info
from .gaps import load_wisdom_adjudications
from .levers import compute_levers, print_levers


def cmd_wisdom_opt(args) -> None:
    if not args.no_rebuild and not CLI.exists():
        cargo_build("fractall-cli")
    adj = load_wisdom_adjudications()
    w = axis_wisdom(not args.quick, runs=args.runs or 3, adjudications=adj,
                    only=args.only)
    card = {"wisdom": w}
    out = BENCH / "wisdom"
    out.mkdir(parents=True, exist_ok=True)
    (out / "wisdom.json").write_text(json.dumps(
        {"date_utc": _now_iso(), "git_sha": git_sha(),
         "machine": machine_info(), **w}, indent=1, ensure_ascii=False) + "\n")
    (out / "wisdom.md").write_text("\n".join(
        [f"# Wisdom — {_now_iso()} (`{git_sha()}`)", ""] + wisdom_md(w)) + "\n")
    print(f"\n✓ rapport : {(out / 'wisdom.md').relative_to(REPO)}")
    print("→ " + " · ".join(f"{n} {k}" for k, n in sorted(w["counts"].items())))
    for g in wisdom_gaps(card, adj):
        print(f"  ⚠ {g['case']} — {g['note']}")
    print_levers(compute_levers(card))
