"""Chemins, registres versionnés, tiers et helpers git/machine du harness."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import os
import platform
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parent.parent.parent

CLI = REPO / "target" / "release" / "fractall-cli"

QUALITY = REPO / "target" / "release" / "fractall-quality"

TOML_DIR = REPO / "toml"

HARNESS_DIR = REPO / "harness"

HISTORY_DIR = HARNESS_DIR / "history"

BASELINE = HARNESS_DIR / "baseline.json"
# Baselines PAR MACHINE (2026-11) : `baselines/<slug-machine>.json`. La
# baseline historique `baseline.json` reste le repli (machine de dev i7). Sans
# ça, une machine secondaire (conteneur, CI, Mac) ne pouvait pas figer SA
# référence sans écraser celle de la machine de dev — donc aucune régression
# intrinsèque n'y était détectable. `machine_slug()` = cpu + nproc normalisés.
BASELINES_DIR = HARNESS_DIR / "baselines"

SCORECARD = REPO / "SCORECARD.md"

BENCH = REPO / "bench" / "harness"

ADJUDICATIONS = HARNESS_DIR / "adjudications.json"

# Adjudications persistées de l'axe wisdom-optimality : case → tradeoff
# correction déjà tranché vs GMP (l'alternative rapide rend une AUTRE image mais
# FAUSSE, le wisdom a raison de router le plus lent-mais-correct). Évite de
# re-litiger le MÊME ADJUDICATE à chaque run — tant que le RÉGIME (routed_path +
# output_differs) est inchangé. Un changement de régime ré-escalade (le verdict
# ADJUDICATE brut est conservé, seul le tag change).
WISDOM_ADJUDICATIONS = HARNESS_DIR / "wisdom-adjudications.json"

# Seed de l'axe fuzz (sondes aléatoires DÉTERMINISTES pert-vs-GMP, cf.
# scripts/fuzz_scenes.py). Committée ici → chaque score rejoue les MÊMES
# scènes ; la faire tourner au rebaseline pour élargir la couverture.
FUZZ_SEED_DEFAULT = 20260714

SCHEMA = 2   # 2 (2026-11) : sections profile, bench, speed.self, trend

SEVERITY_LABEL = {1: "correction", 2: "robustesse", 3: "vitesse", 4: "qualité"}

MARKER = "<!-- généré par scripts/harness.py — ne pas éditer à la main -->"

# --- garde-fou crash / mémoire (voir SKILL improve §Étape 0) ------------------
# Un cas gourmand du corpus peut allouer toute la RAM et faire tomber l'OS
# pendant un sweep étendu. Trois mécanismes évitent que /improve reste coincé :
#   1. INFLIGHT : breadcrumb écrit AVANT chaque cas, effacé après. Un fichier
#      résiduel au démarrage = le cas qui a tué la machine (le process a été
#      emporté avant de pouvoir nettoyer). On le journalise + quarantaine.
#   2. cap mémoire (RLIMIT_AS) : un runaway est tué proprement (loggé) au lieu
#      de faire planter l'OS.
#   3. CRASH_JOURNAL : trace jsonl append-only étudiable de tous les incidents.
INFLIGHT = HARNESS_DIR / "inflight.json"

CRASH_JOURNAL = HARNESS_DIR / "crash-journal.jsonl"

QUARANTINE = HARNESS_DIR / "quarantine.json"

RESOLVED = HARNESS_DIR / "resolved.json"

# Cas où F3 rend une image DÉGÉNÉRÉE (fast-path uniforme faux, cf. glitch_test_5)
# alors que fractall rend la vraie structure. Détecté par l'axe parité
# (compare_f3 status `f3_degenerate`), persisté ici pour que l'axe VITESSE
# (qui peut tourner seul) exclue ces cas du geomean et des gaps : comparer un
# rendu correct à un fast-path faux n'a pas de sens. Auto-entretenu : un run
# parité qui re-classe le cas `ok` le retire.
F3_DEGENERATE = HARNESS_DIR / "f3-degenerate.json"

# Cas CORRECTS et memory-safe mais dont le rendu dépasse le budget TEMPS des
# sweeps (deep-zoom extrême : l'orbite référence GMP single-thread domine —
# ex. e52465 zoom 1e52465, orbite 639k pas × 174kbit ≈ 660s, pic RSS 237MB,
# exit 0 ; F3 aussi hors budget). ATTESTÉ PAR L'OPÉRATEUR (mesure : rendu
# complet exit 0 + pic RSS < cap), PAS auto-détecté — un timeout ne rapporte
# pas de RSS (cf. run_case_measured → peak_rss_kb=None), donc preflight ne peut
# distinguer « lent-mais-sûr » d'un runaway. But : NE PAS conflater lent-safe
# avec crash/OOM. Ces cas sont skippés des sweeps time-bounded (comme la
# quarantaine) MAIS ne produisent PAS de faux gap robustesse-2 (ils rendent
# juste ; hors enveloppe d'excellence perf ≤1e1000). Miroir de F3_DEGENERATE.
SLOW_SAFE = HARNESS_DIR / "slow-safe.json"

# Issues du journal qui prouvent qu'un cas a fait tomber la machine ou a été tué
# sous cap : la quarantaine DOIT les couvrir. On EXCLUT `interrupted` (Ctrl-C
# gracieux), `fail` (rc≠0 bénin — ex. F3 HAVE_EXR=0), `timeout` et `ok`.
HARD_CRASH_OUTCOMES = frozenset(
    {"died_uncleanly", "killed_oom", "aborted", "killed"})

def find_f3() -> Path | None:
    """Localise le binaire Fraktaler-3 — miroir de `compare_f3.find_f3` SANS
    importer compare_f3 (qui exige numpy + OpenEXR et `sys.exit` sinon) : le
    harness doit démarrer sur une machine dépourvue des deps EXR (quality,
    fuzz, goldens, profile, bench restent mesurables). Ordre : env `F3_BIN`,
    `fraktaler-3-3.1.linux`, `fraktaler-3.linux`, `fraktaler-3.macos`."""
    d = REPO / "fraktaler-3-3.1"
    candidates: list[Path] = []
    env = os.environ.get("F3_BIN")
    if env:
        candidates.append(Path(env))
    candidates += [d / "fraktaler-3-3.1.linux", d / "fraktaler-3.linux",
                   d / "fraktaler-3.macos"]
    for c in candidates:
        if c.exists() and os.access(c, os.X_OK) and binary_runs_here(c):
            return c
    return None


def binary_runs_here(path: Path) -> bool:
    """Le binaire est-il du bon format pour CET OS ? (le `.macos` versionné est
    exécutable au sens des droits mais pas lançable sous Linux → l'axe vitesse
    ne doit pas le « trouver », sinon 10 `fail` silencieux). Décision par
    magic bytes : ELF sous Linux, Mach-O sous macOS, PE sous Windows."""
    try:
        magic = path.open("rb").read(4)
    except OSError:
        return False
    if sys.platform.startswith("linux"):
        return magic == b"\x7fELF"
    if sys.platform == "darwin":
        return magic in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe",
                         b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca")
    if sys.platform.startswith("win"):
        return magic[:2] == b"MZ"
    return True


def load_compare_f3():
    """Import PARESSEUX de `scripts/compare_f3.py` (numpy + OpenEXR + PIL).
    Renvoie le module, ou None si ses deps manquent (il `sys.exit` à l'import)
    — les axes speed/parity se marquent alors `f3_deps_missing` au lieu de
    tuer tout le harness."""
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import compare_f3  # noqa: F401
    except SystemExit:
        return None
    except ImportError:
        return None
    return compare_f3


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

# -----------------------------------------------------------------------------

QUICK_CASES = [
    "test5", "spiral", "flake", "glitch_test_2", "e50", "e113", "e401",
    "e1000", "floral_fantasy", "dragon",
]

STANDARD_EXTRA = [
    "e318", "e1121", "e1200", "glitch_test_3", "glitch_test_4", "heaven",
    "integral_of_ex2", "windmill", "mitosis", "golden_spider", "leaded_glass",
    "magic", "tick_tock", "virus", "x",
]

def all_toml_stems() -> list[str]:
    return sorted(p.stem for p in TOML_DIR.glob("*.toml"))

def tier_config(tier: str) -> dict:
    # `quality_width/height` : résolution de l'axe quality (GMP pixel-par-pixel,
    # O(1e3-1e4) plus lent que la perturbation). Le tier `quick` la réduit à 96²
    # (~7× moins de pixels que 256²) pour que le cycle interne reste rapide — les
    # verdicts PASS/WARN/FAIL sont stables en résolution (bord chaotique inclus).
    if tier == "quick":
        return {"cases": list(QUICK_CASES), "width": 256, "height": 256,
                "runs": 1, "timeout": 120.0,
                "quality_width": 96, "quality_height": 96,
                "fuzz_probes": 3}
    if tier == "standard":
        return {"cases": QUICK_CASES + STANDARD_EXTRA, "width": 256,
                "height": 256, "runs": 3, "timeout": 300.0,
                "quality_width": 256, "quality_height": 256,
                "fuzz_probes": 6}
    if tier == "full":
        return {"cases": all_toml_stems(), "width": 256, "height": 256,
                "runs": 1, "timeout": 600.0,
                "quality_width": 256, "quality_height": 256,
                "fuzz_probes": 8}
    raise SystemExit(f"tier inconnu: {tier}")

def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"

def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, text=True
        )
        return bool(out.strip())
    except Exception:
        return False

def machine_info() -> dict:
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    return {"cpu": cpu, "nproc": os.cpu_count(),
            "os": f"{platform.system()} {platform.release()}"}

def machine_slug(machine: dict | None = None) -> str:
    """Identifiant de fichier stable d'une machine : cpu + nproc, minuscules,
    [a-z0-9-] seulement (`intel-r-core-tm-i7-10700f-cpu-2-90ghz-16t`)."""
    m = machine or machine_info()
    raw = f"{m.get('cpu', 'cpu')}-{m.get('nproc', 0)}t".lower()
    slug = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
    return slug or "unknown"


def baseline_path_for(machine: dict | None = None) -> Path:
    """Chemin de la baseline PROPRE à cette machine (existant ou à créer)."""
    return BASELINES_DIR / f"{machine_slug(machine)}.json"


def resolve_baseline(machine: dict | None = None) -> Path | None:
    """Baseline à utiliser pour cette machine : la sienne si elle existe,
    sinon le repli historique `baseline.json`, sinon None."""
    own = baseline_path_for(machine)
    if own.exists():
        return own
    if BASELINE.exists():
        return BASELINE
    return None


def load_baseline(machine: dict | None = None) -> dict | None:
    path = resolve_baseline(machine)
    if path is None:
        return None
    try:
        card = json.loads(path.read_text())
    except Exception:
        return None
    if isinstance(card, dict):
        try:
            card["_file"] = str(path.relative_to(REPO))
        except ValueError:  # baseline hors repo (tests, override)
            card["_file"] = str(path)
    return card if isinstance(card, dict) else None


def cargo_build(bin_name: str | None = None) -> bool:
    cmd = ["cargo", "build", "--release"]
    if bin_name:
        cmd += ["--bin", bin_name]
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=REPO).returncode == 0
