#!/usr/bin/env python3
"""Scorecard orchestrator — mesure fractall vs Fraktaler-3 (protocole HARNESS.md).

Enveloppe mince : la logique vit dans le package `scripts/harnesslib/`
(fichiers < 800 lignes, pilier 5) :

  config.py          chemins, registres versionnés, tiers, helpers git/machine
  guard.py           garde-fou crash/mémoire (inflight, journal, quarantaine…)
  axes_f3.py         axes CONTRE F3 : speed, parity
  axes_truth.py      axes contre la vérité GMP : quality, fuzz, goldens
  gaps.py            gaps triés + régressions vs baseline
  scorecard.py       SCORECARD.md + harness/history
  commands*.py       sous-commandes ; wisdom_opt.py : axe wisdom-optimality
  cli.py             argparse

  python3 scripts/harness.py --help
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harnesslib.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
