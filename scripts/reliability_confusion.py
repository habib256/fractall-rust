#!/usr/bin/env python3
"""Matrice de confusion du détecteur de fiabilité f64 contre la frame dd.

Pour chaque `scène:taille` : rendu `FRACTALL_RELIABILITY=observe` (valeurs
f64 BRUTES + bornes d'erreur par pixel via `FRACTALL_RELIABILITY_DUMP`),
rendu de production (défaut) et juge `--dd-tier` (mis en cache : il ne dépend
pas du traqueur f64). Imprime, par scène :

- `flagged`   pixels dont la borne s'emballe (≥ seuil, 1e30 par défaut) ;
- `true_pos`  flaggés ET faux (valeur f64 ≠ juge dd) ;
- `false_pos` flaggés mais justes (correction dd payée pour rien) ;
- `false_neg` faux mais NON flaggés (sortent faux du rendu de production) ;
- `prod_wrong` pixels ≠ juge dans le rendu de production ;
- `pixels`/`post` : phases du `[PERTURB PERF]` du rendu de production.

Les comptes sont déterministes ; les temps ne le sont pas (machine partagée).
Toute variable d'environnement est transmise aux trois rendus sauf
`FRACTALL_RELIABILITY*` (imposée par rendu) : `FRACTALL_RELIABILITY_RUNAWAY=1e8
python3 scripts/reliability_confusion.py …` mesure un autre seuil.

Usage :
    bench/venv/bin/python scripts/reliability_confusion.py e50:256 dragon:128 …
    (--obs-only : saute le rendu de production)
"""

from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harnesslib.axes_precision import read_fmap_iterations  # noqa: E402
from harnesslib.config import BENCH, CLI, TOML_DIR  # noqa: E402

OUT = BENCH / "reliability_confusion"
PERF_RE = re.compile(r"\[PERTURB PERF\] \d+x\d+ .*?pixels=([\d.]+)s post=([\d.]+)s total=([\d.]+)s")
PATH_RE = re.compile(r"\[FRACTALL\][^\n]*?path=(\S+)")


def render(scene: str, size: int, out: Path, extra: list[str], env: dict[str, str]) -> str:
    e = {k: v for k, v in os.environ.items() if not k.startswith("FRACTALL_RELIABILITY")}
    e["FRACTALL_PERTURB_STATS"] = "1"
    e.update(env)
    cmd = [str(CLI), "--toml", str(TOML_DIR / f"{scene}.toml"), "--width", str(size),
           "--height", str(size), "--no-gpu", "--output", str(out.with_suffix(".png")),
           "--output-map", str(out)] + extra
    p = subprocess.run(cmd, env=e, capture_output=True, text=True, timeout=3600)
    if p.returncode != 0:
        raise RuntimeError(f"{scene}:{size} rc={p.returncode}\n{p.stderr[-2000:]}")
    return p.stderr + p.stdout


def measure(scene: str, size: int, obs_only: bool) -> dict:
    (OUT / "judge").mkdir(parents=True, exist_ok=True)
    judge = OUT / "judge" / f"{scene}_{size}.fmap"
    if not judge.exists():
        # Juge : aucun pixel dd non fiable laissé sans recalcul GMP.
        render(scene, size, judge, ["--dd-tier"],
               {"FRACTALL_RELIABILITY_GMP_MAX_FRACTION": "1"})
    truth = read_fmap_iterations(judge)
    obs = OUT / f"{scene}_{size}_obs.fmap"
    dump = OUT / f"{scene}_{size}.f32"
    runaway = float(os.environ.get("FRACTALL_RELIABILITY_RUNAWAY", "1e30"))
    log_o = render(scene, size, obs, [], {
        "FRACTALL_RELIABILITY": "observe", "FRACTALL_RELIABILITY_DUMP": str(dump),
        "FRACTALL_RELIABILITY_RUNAWAY": str(runaway)})
    f64 = read_fmap_iterations(obs)
    if dump.exists():
        raw = dump.read_bytes()
        bounds = struct.unpack(f"<{len(raw) // 4}f", raw)
    else:  # path non suivi (std, hybride…) : rien de flaggé
        bounds = (0.0,) * len(f64)
    flag = [not (b < runaway) for b in bounds]
    wrong = [a != b for a, b in zip(f64, truth)]
    tp = sum(f and w for f, w in zip(flag, wrong))
    fn = sum(w and not f for f, w in zip(flag, wrong))
    row = dict(scene=scene, size=size, flagged=sum(flag), true_pos=tp,
               false_pos=sum(flag) - tp, f64_wrong=sum(wrong), false_neg=fn)
    log = log_o
    if not obs_only:
        prod = OUT / f"{scene}_{size}_prod.fmap"
        log = render(scene, size, prod, [], {})
        row["prod_wrong"] = sum(a != b for a, b in zip(read_fmap_iterations(prod), truth))
    perf = PERF_RE.findall(log)
    path = PATH_RE.findall(log)
    if perf:
        row["pixels_s"], row["post_s"], row["total_s"] = (float(x) for x in perf[-1])
    row["path"] = path[-1] if path else "?"
    return row


def main(argv: list[str]) -> int:
    obs_only = "--obs-only" in argv
    cases = [a for a in argv if not a.startswith("--")]
    if not cases or not CLI.exists():
        print(__doc__)
        return 2
    rows = []
    for c in cases:
        scene, size = c.split(":")
        r = measure(scene, int(size), obs_only)
        rows.append(r)
        print(f"{scene:>16}:{size:<4} flagged={r['flagged']:6} vrais={r['true_pos']:6} "
              f"faux_pos={r['false_pos']:6} faux_neg={r['false_neg']:4} "
              f"prod≠dd={r.get('prod_wrong', '-')!s:>4} post={r.get('post_s', '-')} "
              f"path={r['path']}", flush=True)
    (OUT / "last.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
