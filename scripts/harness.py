#!/usr/bin/env python3
"""Scorecard orchestrator — mesure fractall vs Fraktaler-3 (protocole HARNESS.md).

Produit `harness/history/<UTC>-<sha>.json` + `SCORECARD.md` (gaps triés) sur 4
axes : speed (ratio wall-clock vs F3), parity (compare_f3.py), quality
(fractall-quality suite vs GMP), goldens (cargo test golden_images).

  score [--tier quick|standard|full] [--axes ...] [--no-rebuild]
        [--cases a,b,c] [--width N] [--height N] [--runs N] [--timeout S]
  baseline   # fige le dernier history comme baseline
  gaps       # ré-affiche les gaps du dernier history

F3 auto-détecté (compare_f3.find_f3, override F3_BIN) ; sans lui speed/parity
sont `f3_unavailable` et le reste tourne.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_f3  # noqa: E402  (réutilise find_f3 / write_f3_wrapper / parse_light_toml)

REPO = Path(__file__).resolve().parent.parent
CLI = REPO / "target" / "release" / "fractall-cli"
QUALITY = REPO / "target" / "release" / "fractall-quality"
TOML_DIR = REPO / "toml"
HARNESS_DIR = REPO / "harness"
HISTORY_DIR = HARNESS_DIR / "history"
BASELINE = HARNESS_DIR / "baseline.json"
SCORECARD = REPO / "SCORECARD.md"
BENCH = REPO / "bench" / "harness"
ADJUDICATIONS = HARNESS_DIR / "adjudications.json"

# Seed de l'axe fuzz (sondes aléatoires DÉTERMINISTES pert-vs-GMP, cf.
# scripts/fuzz_scenes.py). Committée ici → chaque score rejoue les MÊMES
# scènes ; la faire tourner au rebaseline pour élargir la couverture.
FUZZ_SEED_DEFAULT = 20260714

SCHEMA = 1
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

try:
    import resource  # POSIX only
except ImportError:  # pragma: no cover
    resource = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_mem_limit_mb() -> int:
    """Cap mémoire par process. Défaut = 85 % de MemTotal (0 = désactivé).

    Override : env `FRACTALL_HARNESS_MEM_MB` ou `--mem-limit-mb`. But : un cas
    emballé est tué par le noyau (alloc échoue → abort, loggé) au lieu de
    thrasher le swap et bloquer l'OS. 85 % laisse tourner les cas GMP lourds
    légitimes ; un faux positif éventuel est visible dans le journal, pas un
    plantage silencieux.
    """
    env = os.environ.get("FRACTALL_HARNESS_MEM_MB")
    if env is not None:
        try:
            return int(env)
        except ValueError:
            pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(int(line.split()[1]) / 1024 * 0.85)
    except Exception:
        pass
    return 0


def _rlimit_preexec(mem_mb: int):
    """preexec_fn posant RLIMIT_AS sur l'enfant (None si cap off/non-POSIX)."""
    if not mem_mb or resource is None:
        return None
    limit = mem_mb * 1024 * 1024

    def _apply():
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    return _apply


def _classify_rc(rc: int) -> str:
    """Statut d'un process terminé. rc<0 = tué par signal -rc."""
    if rc == 0:
        return "ok"
    if rc >= 0:
        return "fail"
    sig = -rc
    if sig == 9:
        return "killed_oom"     # SIGKILL — OOM-killer du noyau
    if sig == 6:
        return "aborted"        # SIGABRT — alloc échouée sous cap, ou panic=abort
    return "killed"             # autre signal


def _append_journal(rec: dict) -> None:
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    with CRASH_JOURNAL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def journal_begin(phase: str, case: str, extra: dict | None = None) -> dict:
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    rec = {"phase": phase, "case": case, "started_utc": _now_iso(),
           "pid": os.getpid(), "git_sha": git_sha()}
    if extra:
        rec.update(extra)
    INFLIGHT.write_text(json.dumps(rec, ensure_ascii=False))
    return rec


def journal_end(rec: dict, outcome: str, extra: dict | None = None) -> None:
    rec = dict(rec)
    rec["outcome"] = outcome
    rec["ended_utc"] = _now_iso()
    if extra:
        rec.update({k: v for k, v in extra.items() if v is not None})
    # incident (≠ ok/timeout attendu) → trace permanente ; sinon on efface juste
    if outcome not in ("ok",):
        _append_journal(rec)
    INFLIGHT.unlink(missing_ok=True)


def load_quarantine() -> dict:
    if not QUARANTINE.exists():
        return {}
    try:
        d = json.loads(QUARANTINE.read_text())
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save_quarantine(d: dict) -> None:
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    QUARANTINE.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")


def load_f3_degenerate() -> dict:
    if not F3_DEGENERATE.exists():
        return {}
    try:
        d = json.loads(F3_DEGENERATE.read_text())
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def load_slow_safe() -> dict:
    if not SLOW_SAFE.exists():
        return {}
    try:
        d = json.loads(SLOW_SAFE.read_text())
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save_slow_safe(d: dict) -> None:
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    SLOW_SAFE.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")


def update_f3_degenerate(parity_cases: dict) -> None:
    """Réconcilie le registre avec le dernier run parité : ajoute les cas
    classés `f3_degenerate`, retire ceux re-mesurés `ok` (F3 corrigé/re-testé).
    Les cas non couverts par ce run restent tels quels."""
    d = load_f3_degenerate()
    changed = False
    for name, c in parity_cases.items():
        if c.get("status") == "f3_degenerate":
            if name not in d:
                d[name] = {"added_utc": _now_iso(),
                           "note": "F3 uniforme (fast-path faux), fractall "
                                   "structuré — cf. bench/harness/parity"}
                changed = True
        elif c.get("status") == "ok" and name in d:
            del d[name]
            changed = True
    if changed:
        HARNESS_DIR.mkdir(parents=True, exist_ok=True)
        F3_DEGENERATE.write_text(
            json.dumps(d, indent=2, ensure_ascii=False) + "\n")


def load_resolved() -> dict:
    """Tombstones {case: resolved_utc} : cas retirés MANUELLEMENT de la
    quarantaine (fix vérifié). La réconciliation journal→quarantaine les
    respecte tant qu'aucun incident PLUS RÉCENT n'apparaît."""
    if not RESOLVED.exists():
        return {}
    try:
        d = json.loads(RESOLVED.read_text())
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save_resolved(d: dict) -> None:
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    RESOLVED.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")


def add_quarantine(case: str, reason: str | None) -> bool:
    d = load_quarantine()
    # une (re)mise en quarantaine périme un éventuel tombstone « résolu » :
    # un nouvel incident supersede le vouching manuel précédent.
    res = load_resolved()
    if case in res:
        del res[case]
        save_resolved(res)
    if case in d:
        return False
    d[case] = {"reason": reason or "manuel", "added_utc": _now_iso()}
    save_quarantine(d)
    return True


def remove_quarantine(case: str) -> bool:
    d = load_quarantine()
    if case not in d:
        return False
    del d[case]
    save_quarantine(d)
    # tombstone : l'opérateur atteste le fix (protocole /improve : remove UNIQUEMENT
    # une fois le fix vérifié) → la réconciliation ne le re-quarantaine pas, sauf
    # nouvel incident postérieur.
    res = load_resolved()
    res[case] = _now_iso()
    save_resolved(res)
    return True


def _incident_ts(rec: dict) -> str:
    return (rec.get("detected_utc") or rec.get("ended_utc")
            or rec.get("started_utc") or "")


def latest_hard_crashes() -> dict:
    """Parcourt le journal append-only et renvoie, par cas, l'horodatage du
    dernier incident PROUVANT un crash/OOM (cf. HARD_CRASH_OUTCOMES)."""
    out: dict[str, str] = {}
    if not CRASH_JOURNAL.exists():
        return out
    for ln in CRASH_JOURNAL.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)
        except Exception:
            continue
        if rec.get("outcome") not in HARD_CRASH_OUTCOMES:
            continue
        # Seuls les crashs du binaire FRACTALL justifient la quarantaine (elle
        # protège la machine des sweeps fractall). Un abort/OOM CÔTÉ F3
        # (phase `speed_f3` — ex. seahorse 1e1392 : F3 sig 6, fractall rend en
        # 119 s) ne doit PAS exclure le cas ; il apparaît déjà comme
        # `f3_aborted`/`f3_timeout` dans l'axe vitesse.
        if rec.get("phase") == "speed_f3":
            continue
        case = rec.get("case")
        if not case or case == "?":
            continue
        ts = _incident_ts(rec)
        if ts >= out.get(case, ""):
            out[case] = ts
    return out


def reconcile_quarantine_from_journal() -> list[str]:
    """Auto-répare l'invariant « crash journalisé ⇒ quarantainé ».

    Le journal (append-only, durable) est la vérité de terrain ; `quarantine.json`
    est versionné et peut DÉRIVER (revert git, reset, checkout) → un cas qui a
    tué la machine peut se retrouver hors quarantaine et rejoindre un sweep
    (cas réel : e22522 died_uncleanly resté loose). On re-quarantaine tout
    cas hard-crash non couvert, SAUF s'il a été résolu manuellement APRÈS son
    dernier incident. Renvoie la liste des cas re-quarantainés."""
    quarantined = set(load_quarantine())
    resolved = load_resolved()
    readded: list[str] = []
    for case, ts in sorted(latest_hard_crashes().items()):
        if case in quarantined:
            continue
        res_ts = resolved.get(case)
        if res_ts is not None and res_ts >= ts:
            continue  # fix vouché après le dernier crash → on respecte
        if add_quarantine(case, f"réconcilié depuis le journal (incident "
                                 f"{ts or '?'} non couvert)"):
            readded.append(case)
    if readded:
        print(f"⚠️  quarantaine réconciliée depuis le journal : "
              f"{', '.join(readded)} (crash journalisé, hors quarantaine → "
              f"ré-exclu des sweeps). Étudier via `preflight`, puis "
              f"`quarantine remove` une fois le fix vérifié.")
    return readded


def check_stale_inflight(auto_quarantine: bool = True) -> str | None:
    """Détecte un breadcrumb résiduel = crash non nettoyé du run précédent.

    Journalise l'incident, met le cas fautif en quarantaine (skip auto au
    prochain sweep), efface le breadcrumb. Retourne le nom du cas ou None.
    """
    if not INFLIGHT.exists():
        return None
    try:
        rec = json.loads(INFLIGHT.read_text())
    except Exception:
        rec = {"case": "?", "phase": "?"}
    case = rec.get("case", "?")
    rec = dict(rec)
    rec["outcome"] = "died_uncleanly"
    rec["detected_utc"] = _now_iso()
    rec["note"] = ("run précédent mort sans nettoyage propre (OOM / crash OS "
                   "probable) — ce cas tournait au moment du plantage")
    _append_journal(rec)
    INFLIGHT.unlink(missing_ok=True)
    print("\n" + "=" * 72)
    print("⚠️  CRASH DÉTECTÉ — le run précédent est mort pendant un cas.")
    print(f"    cas en vol : {case!r}  (phase {rec.get('phase')}, "
          f"{rec.get('size', '?')})")
    print(f"    → incident journalisé dans "
          f"{CRASH_JOURNAL.relative_to(REPO)}")
    if auto_quarantine and case and case != "?":
        if add_quarantine(case, rec["note"]):
            print(f"    → {case!r} mis en QUARANTAINE (skip auto des sweeps).")
        print("    → étudier via `python3 scripts/harness.py quarantine list` "
              "puis `preflight`.")
    print("=" * 72 + "\n")
    return case


def _on_interrupt(signum, _frame):
    """SIGINT (Ctrl-C) / SIGTERM : terminaison GRACIEUSE → nettoyer le breadcrumb.

    Sans ça, interrompre un sweep laisse `inflight.json` en place → le run
    suivant le prend pour un crash et QUARANTAINE à tort le cas en vol (le cas
    n'a pas planté, l'utilisateur a coupé). Une mort NON catchable (SIGKILL,
    panne OS, OOM-killer) ne passe PAS ici et laisse le breadcrumb → toujours
    détectée comme `died_uncleanly` : c'est exactement la distinction voulue.
    """
    if INFLIGHT.exists():
        try:
            rec = json.loads(INFLIGHT.read_text())
        except Exception:
            rec = {}
        rec = dict(rec)
        rec["outcome"] = "interrupted"
        rec["ended_utc"] = _now_iso()
        rec["signal"] = signum
        _append_journal(rec)          # trace, mais NE déclenche PAS de quarantaine
        INFLIGHT.unlink(missing_ok=True)
        print(f"\n⚠️  interruption (signal {signum}) — breadcrumb nettoyé "
              f"(pas de fausse quarantaine).", flush=True)
    raise SystemExit(128 + signum)


def install_signal_handlers() -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _on_interrupt)
        except (ValueError, OSError):
            pass  # hors thread principal — best effort
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

def cargo_build(bin_name: str | None = None) -> bool:
    cmd = ["cargo", "build", "--release"]
    if bin_name:
        cmd += ["--bin", bin_name]
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=REPO).returncode == 0

def timed_runs(cmd: list[str], env: dict, timeout: float, runs: int,
               accept_nonzero: bool = False,
               mem_limit_mb: int = 0) -> tuple[str, float | None]:
    """Chronomètre `cmd` `runs` fois (médiane). Renvoie (status, secondes|None).

    status : ok | timeout | fail | killed_oom | aborted | killed.
    `accept_nonzero=True` enregistre quand même le temps si le process va au
    bout avec rc != 0 (cas F3 batch sans EXR : rendu complet mais save_exr
    no-op → rc != 0). `mem_limit_mb` pose RLIMIT_AS sur l'enfant (garde-fou
    anti-plantage-OS) ; un runaway est alors tué (rc<0) au lieu de saturer la
    RAM — statut `killed_oom`/`aborted` propagé au caller.
    """
    times: list[float] = []
    preexec = _rlimit_preexec(mem_limit_mb)
    for _ in range(runs):
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env=env, timeout=timeout, preexec_fn=preexec,
            )
        except subprocess.TimeoutExpired:
            return "timeout", None
        except OSError:
            # binaire non exécutable ici (ex: build .macos lancé sur Linux)
            return "fail", None
        rc = proc.returncode
        if rc < 0:
            return _classify_rc(rc), None
        if rc != 0 and not accept_nonzero:
            return "fail", None
        times.append(time.monotonic() - t0)
    return "ok", statistics.median(times)

def axis_speed(cases: list[str], width: int, height: int, runs: int,
               timeout: float, f3_bin: Path | None,
               mem_limit_mb: int = 0, quarantined: set | None = None,
               slow_safe: set | None = None) -> dict:
    outdir = BENCH / "speed"
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="harness_speed_"))
    results: dict[str, dict] = {}
    quarantined = quarantined or set()
    slow_safe = slow_safe or set()
    fr_env = os.environ.copy()
    fr_env["FRACTALL_NO_AUTO_ADJUST"] = "1"
    fr_env["FRACTALL_NO_PERIOD"] = "1"
    try:
        for i, name in enumerate(cases, 1):
            toml_path = TOML_DIR / f"{name}.toml"
            entry = {"fractall_s": None, "f3_s": None, "ratio": None,
                     "status": "ok", "f3_exr_ok": False}
            print(f"  [{i:>3}/{len(cases)}] speed {name:24s}", end=" ", flush=True)
            if name in quarantined:
                entry["status"] = "quarantined"
                results[name] = entry
                print("⊘ QUARANTAINE (skip — voir quarantine.json)")
                continue
            if name in slow_safe:
                # correct + memory-safe mais hors budget temps → skip SANS gap
                # robustesse (ratio None, exclu des agrégats comme f3_degenerate).
                entry["status"] = "slow_safe"
                results[name] = entry
                print("⊘ SLOW-SAFE (skip — correct mais hors budget, "
                      "voir slow-safe.json)")
                continue
            if not toml_path.exists():
                entry["status"] = "missing_toml"
                results[name] = entry
                print("✗ toml absent")
                continue

            fr_cmd = [str(CLI), "--toml", str(toml_path), "--width", str(width),
                      "--height", str(height), "--bailout", "25",
                      "--output", str(outdir / f"{name}.png")]
            jrec = journal_begin("speed", name,
                                 {"size": f"{width}x{height}",
                                  "toml": str(toml_path)})
            st_fr, med_fr = timed_runs(fr_cmd, fr_env, timeout, runs,
                                       mem_limit_mb=mem_limit_mb)
            journal_end(jrec, st_fr, {"secs": med_fr})
            if st_fr == "ok":
                entry["fractall_s"] = round(med_fr, 4)
            elif st_fr == "timeout":
                entry["status"] = "fractall_timeout"
            elif st_fr in ("killed_oom", "aborted", "killed"):
                entry["status"] = f"fractall_{st_fr}"
                # gourmand mémoire tué par le cap → quarantaine auto pour ne pas
                # replanter le sweep suivant ; l'incident reste dans le journal.
                if add_quarantine(name, f"{st_fr} sous cap mémoire "
                                        f"{mem_limit_mb}MB @ {width}x{height}"):
                    print("⚠ tué (mémoire) → QUARANTAINE  ", end="")
            else:
                entry["status"] = "fractall_fail"

            if f3_bin is None:
                # F3 absent : on garde le temps fractall, statut f3_unavailable
                if entry["status"] == "ok":
                    entry["status"] = "f3_unavailable"
                results[name] = entry
                fs = entry["fractall_s"]
                print(f"⊘ fr={fs}s (F3 indispo)")
                continue

            src = compare_f3.parse_light_toml(toml_path)
            iters = src.iterations or 1024
            if iters > 2 ** 31 - 1:
                iters = 2 ** 31 - 1
            wrapper = compare_f3.write_f3_wrapper(
                src, tmp, name, width, height, iters, 25.0)
            f3_exr = tmp / f"{name}_f3.exr"
            if f3_exr.exists():
                f3_exr.unlink()
            f3_cmd = [str(f3_bin), "-b", "-P", str(wrapper)]
            # accept_nonzero : le build Linux (HAVE_EXR=0) rend complètement mais
            # exit rc != 0 (save_exr no-op) — on garde le timing si pas timeout.
            jrec_f3 = journal_begin("speed_f3", name,
                                    {"size": f"{width}x{height}"})
            st_f3, med_f3 = timed_runs(f3_cmd, os.environ.copy(), timeout, runs,
                                       accept_nonzero=True,
                                       mem_limit_mb=mem_limit_mb)
            journal_end(jrec_f3, st_f3, {"secs": med_f3})
            entry["f3_exr_ok"] = f3_exr.exists()
            if st_f3 == "ok":
                entry["f3_s"] = round(med_f3, 4)
            elif st_f3 == "timeout" and entry["status"] == "ok":
                entry["status"] = "f3_timeout"
            elif st_f3 in ("fail", "killed_oom", "aborted", "killed") \
                    and entry["status"] == "ok":
                entry["status"] = "f3_fail" if st_f3 == "fail" else f"f3_{st_f3}"

            if entry["fractall_s"] and entry["f3_s"]:
                entry["ratio"] = round(entry["fractall_s"] / entry["f3_s"], 4)
            results[name] = entry
            print(f"fr={entry['fractall_s']}s f3={entry['f3_s']}s "
                  f"ratio={entry['ratio']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # Exclure les cas F3-dégénérés des AGRÉGATS (geomean/pire/wins) : F3 y rend
    # un fast-path uniforme FAUX (cf. registre f3-degenerate.json, entretenu par
    # l'axe parité) → le ratio mesuré est un artefact, pas un gap moteur
    # (glitch_test_5 3.8× fantôme, 2026-07-12). Le ratio mesuré reste stocké
    # dans l'entrée (flag `f3_degenerate`) et listé à part dans le scorecard.
    degen = set(load_f3_degenerate())
    for n, e in results.items():
        if n in degen:
            e["f3_degenerate"] = True
    excluded_degen = [n for n in results if n in degen
                      and results[n]["ratio"] is not None]
    ratios = [e["ratio"] for n, e in results.items()
              if e["ratio"] is not None and n not in degen]
    geomean = (math.exp(sum(math.log(r) for r in ratios) / len(ratios))
               if ratios else None)
    worst = None
    worst_pairs = [(n, e["ratio"]) for n, e in results.items()
                   if e["ratio"] is not None and n not in degen]
    if worst_pairs:
        n, r = max(worst_pairs, key=lambda x: x[1])
        worst = {"case": n, "ratio": round(r, 4)}
    wins = [n for n, e in results.items()
            if e["ratio"] is not None and e["ratio"] < 1.0 and n not in degen]
    timeouts = [n for n, e in results.items()
                if e["status"] in ("fractall_timeout", "f3_timeout")]
    return {
        "status": "ok" if f3_bin else "f3_unavailable",
        "cases": results,
        "geomean_ratio": round(geomean, 4) if geomean else None,
        "worst_ratio": worst,
        "wins": wins,
        "timeouts": timeouts,
        "excluded_f3_degenerate": excluded_degen,
        "excluded_slow_safe": sorted(
            n for n, e in results.items() if e["status"] == "slow_safe"),
        "n_cases": len(cases),
        "n_compared": len(ratios),
    }

def _fnum(row: dict, key: str):
    v = row.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None

def axis_parity(cases: list[str], width: int, height: int, timeout: float,
                f3_bin: Path | None, mem_limit_mb: int = 0) -> dict:
    empty = {"cases": {}, "n_ok": 0, "n_pixel_equiv": 0, "n_fail": 0,
             "n_timeout": 0, "n_f3_degenerate": 0}
    if f3_bin is None:
        return {"status": "f3_unavailable", **empty}
    outdir = BENCH / "parity"
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(REPO / "scripts" / "compare_f3.py"),
           "--only", ",".join(cases), "--width", str(width),
           "--height", str(height), "--timeout", str(timeout),
           "--out", str(outdir)]
    print(f"  $ compare_f3.py --only {len(cases)} cas", flush=True)
    jrec = journal_begin("parity", "<compare_f3-suite>",
                         {"size": f"{width}x{height}", "n_cases": len(cases)})
    proc = subprocess.run(cmd, cwd=REPO, preexec_fn=_rlimit_preexec(mem_limit_mb))
    journal_end(jrec, _classify_rc(proc.returncode))
    csv_path = outdir / "_summary.csv"
    if not csv_path.exists():
        return {"status": "error", **empty}

    cases_out: dict[str, dict] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            name = row.get("name", "?")
            cases_out[name] = {
                "status": row.get("status", "?"),
                "rel_dsi_pct": _fnum(row, "rel_dsi_pct"),
                "mean_abs_dsi": _fnum(row, "mean_abs_dsi"),
                "inside_mismatch": int(row["inside_mismatch"])
                if row.get("inside_mismatch") else None,
                "f3_degenerate": row.get("status") == "f3_degenerate",
                "f3_secs_num": _fnum(row, "f3_secs_num"),
                "fr_secs_num": _fnum(row, "fr_secs_num"),
            }

    def st(name):
        return cases_out[name]["status"]

    n_ok = sum(1 for n in cases_out if st(n) == "ok")
    n_pixel_equiv = sum(
        1 for n, c in cases_out.items()
        if c["status"] == "ok" and c["rel_dsi_pct"] is not None
        and c["rel_dsi_pct"] < 0.01)
    n_timeout = sum(1 for n in cases_out
                    if st(n) in ("fractall_timeout", "f3_timeout"))
    n_degen = sum(1 for n in cases_out if st(n) == "f3_degenerate")
    n_fail = sum(1 for n in cases_out if st(n) not in (
        "ok", "f3_degenerate", "fractall_timeout", "f3_timeout"))
    # Heuristique : F3 sans support EXR (build Linux HAVE_EXR=0) → tous les cas
    # tombent en f3_fail (aucun .exr produit). On le signale distinctement pour
    # ne pas confondre avec un vrai bug de rendu (rebuild EXR en cours).
    statuses = [c["status"] for c in cases_out.values()]
    axis_status = "ok"
    if (statuses and n_ok == 0 and n_degen == 0
            and all(x in ("f3_fail", "f3_timeout") for x in statuses)
            and any(x == "f3_fail" for x in statuses)):
        axis_status = "f3_no_exr"
    # Persister les cas F3-dégénérés pour l'axe vitesse (qui peut tourner seul) :
    # leurs ratios n'ont pas de sens (F3 fast-path faux vs rendu correct).
    if axis_status == "ok":
        update_f3_degenerate(cases_out)
    return {
        "status": axis_status,
        "cases": cases_out,
        "n_ok": n_ok,
        "n_pixel_equiv": n_pixel_equiv,
        "n_fail": n_fail,
        "n_timeout": n_timeout,
        "n_f3_degenerate": n_degen,
    }

def _qnum(it: dict, *keys):
    for k in keys:
        if k in it and it[k] is not None:
            try:
                return float(it[k])
            except (TypeError, ValueError):
                pass
    return None

def parse_quality_json(path: Path) -> list[dict] | None:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    items = None
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for k in ("presets", "rows", "results", "suite"):
            if isinstance(data.get(k), list):
                items = data[k]
                break
        if items is None and data and all(
                isinstance(v, dict) for v in data.values()):
            items = [{**v, "name": k} for k, v in data.items()]
    if items is None:
        return None
    rows = []
    for it in items:
        if not isinstance(it, dict):
            continue
        name = it.get("name") or it.get("preset") or "?"
        verdict = str(it.get("verdict") or it.get("status") or "").upper()
        rows.append({
            "name": name, "verdict": verdict,
            "max_iter_diff": _qnum(it, "max_iter_diff", "max"),
            "p99_iter_diff": _qnum(it, "p99_iter_diff", "p99"),
            "divergence_ratio": _qnum(it, "divergence_ratio",
                                      "iter_divergence_ratio"),
            "time_pert_ms": _qnum(it, "time_pert_ms", "perturb_time_ms"),
            "time_gmp_ms": _qnum(it, "time_gmp_ms", "gmp_time_ms"),
        })
    return rows or None

def parse_quality_md(text: str) -> list[dict] | None:
    """Fallback : colonnes de report.rs::write_suite_summary.

    | Preset | Verdict | max_iter_diff | p99_iter_diff | divergence_ratio |
      escape_disagree | max_|dz| | time_pert_ms | time_gmp_ms | speedup |
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 10:
            continue
        m = re.match(r"\[([^\]]+)\]", cells[0])
        if not m:  # entête / séparateur
            continue
        rows.append({
            "name": m.group(1),
            "verdict": cells[1].strip("* "),
            "max_iter_diff": _safe_float(cells[2]),
            "p99_iter_diff": _safe_float(cells[3]),
            "divergence_ratio": _safe_float(cells[4]),
            "time_pert_ms": _safe_float(cells[7]),
            "time_gmp_ms": _safe_float(cells[8]),
        })
    return rows or None

def _safe_float(s: str):
    try:
        return float(s.rstrip("x"))
    except (ValueError, AttributeError):
        return None

def summarize_quality(rows: list[dict]) -> dict:
    def has(r, tag):
        return r["verdict"].upper().startswith(tag)
    n_pass = sum(1 for r in rows if has(r, "PASS"))
    n_warn = sum(1 for r in rows if has(r, "WARN"))
    n_fail = sum(1 for r in rows if has(r, "FAIL"))
    return {
        "status": "ok",
        "n_pass": n_pass, "n_warn": n_warn, "n_fail": n_fail,
        "fail_presets": [r["name"] for r in rows if has(r, "FAIL")],
        "warn_presets": [r["name"] for r in rows if has(r, "WARN")],
        "presets": rows,
    }

def axis_quality(no_rebuild: bool, width: int = 256, height: int = 256,
                 mem_limit_mb: int = 0) -> dict:
    if not QUALITY.exists() and not no_rebuild:
        cargo_build("fractall-quality")
    if not QUALITY.exists():
        # tentative de build même en --no-rebuild : sans binaire, rien à faire
        cargo_build("fractall-quality")
    if not QUALITY.exists():
        return {"status": "no_binary"}
    outdir = BENCH / "quality"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"  $ fractall-quality suite --width {width} --height {height}", flush=True)
    jrec = journal_begin("quality", "<quality-suite>",
                         {"size": f"{width}x{height}"})
    proc = subprocess.run([str(QUALITY), "suite", "--output-dir", str(outdir),
                           "--width", str(width), "--height", str(height)],
                          cwd=REPO, preexec_fn=_rlimit_preexec(mem_limit_mb))
    journal_end(jrec, _classify_rc(proc.returncode))
    rows = None
    jf = outdir / "suite-summary.json"
    if jf.exists():
        rows = parse_quality_json(jf)
    if rows is None:
        md = outdir / "suite-summary.md"
        if md.exists():
            rows = parse_quality_md(md.read_text())
    if rows is None:
        return {"status": "parse_error", "returncode": proc.returncode}
    return summarize_quality(rows)

def axis_goldens() -> dict:
    print("  $ cargo test --release --test golden_images", flush=True)
    proc = subprocess.run(
        ["cargo", "test", "--release", "--test", "golden_images"],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    passed = proc.returncode == 0
    detail = "" if passed else "\n".join(proc.stdout.splitlines()[-40:])
    return {"passed": passed, "detail": detail}

def axis_fuzz(probes: int, seed: int, width: int = 96, height: int = 96,
              mem_limit_mb: int = 0) -> dict:
    """Sondes aléatoires DÉTERMINISTES pert-vs-GMP (G8.2). Les scènes sont des
    points frontière stables (échappement invariant à un bump 2⁻⁴⁶ de c —
    ground truth bien conditionnée, cf. scripts/fuzz_scenes.py) tirés sur les
    6 familles Mandelbrot-like, zoom 1e5-1e11. Vise les classes de divergence
    que les presets figés ratent (ex. e13/dd-sensibilité, trouvée par accident).
    Seed committée dans le scorecard → reproductible."""
    if not QUALITY.exists():
        cargo_build("fractall-quality")
    if not QUALITY.exists():
        return {"status": "no_binary"}
    sys.path.insert(0, str(REPO / "scripts"))
    import fuzz_scenes
    outdir = BENCH / "fuzz"
    outdir.mkdir(parents=True, exist_ok=True)
    # Cache des scènes par (seed, n) : la génération coûte ~5 s/scène.
    cache = outdir / f"scenes-{seed}-{probes}.json"
    if cache.exists():
        scenes = json.loads(cache.read_text())
    else:
        print(f"  génération de {probes} scènes fuzz (seed {seed})…", flush=True)
        scenes = fuzz_scenes.generate(seed, probes)
        cache.write_text(json.dumps(scenes, indent=1))
    cases: dict[str, dict] = {}
    for s in scenes:
        name = s["name"]
        print(f"  $ fractall-quality compare --type {s['type_id']} "
              f"--zoom {s['zoom']} --name {name}", flush=True)
        jrec = journal_begin("fuzz", name, {"size": f"{width}x{height}"})
        proc = subprocess.run(
            [str(QUALITY), "compare", "--type", str(s["type_id"]),
             f"--center-x-hp={s['center_x']}",
             f"--center-y-hp={s['center_y']}",
             "--zoom", s["zoom"], "--iterations", str(s["iterations"]),
             "--width", str(width), "--height", str(height),
             "--name", name, "--output-dir", str(outdir)],
            cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            preexec_fn=_rlimit_preexec(mem_limit_mb))
        journal_end(jrec, _classify_rc(proc.returncode))
        verdict = "ERROR"
        rj = outdir / name / "report.json"
        if proc.returncode == 0 and rj.exists():
            try:
                verdict = json.loads(rj.read_text()).get("verdict") or "ERROR"
            except (json.JSONDecodeError, OSError):
                pass
        cases[name] = {"verdict": verdict, "type": s["type"],
                       "center_x": s["center_x"], "center_y": s["center_y"],
                       "zoom": s["zoom"], "iterations": s["iterations"]}
    verdicts = [c["verdict"] for c in cases.values()]
    return {
        "status": "ok", "seed": seed, "n_probes": len(scenes),
        "n_pass": verdicts.count("PASS"),
        "n_warn": verdicts.count("WARN"),
        "n_fail": len(verdicts) - verdicts.count("PASS") - verdicts.count("WARN"),
        "cases": cases,
        "fail_cases": [k for k, v in cases.items()
                       if v["verdict"] not in ("PASS", "WARN")],
        "warn_cases": [k for k, v in cases.items() if v["verdict"] == "WARN"],
    }

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
    # --- régressions vs baseline (>10%) ---
    if base and _comparable(card, base):
        gaps += _regression_gaps(card, base)
    return gaps

def _regression_gaps(card: dict, base: dict) -> list[dict]:
    out = []
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
    if base and not comparable:
        L.append(f"- _baseline présente mais tier différent "
                 f"({base.get('meta', {}).get('tier')}) — pas de delta._")
    L.append("")

    # Speed
    s = card.get("speed")
    if s and s.get("status") != "skipped":
        L += ["## Vitesse (ratio fractall/F3, <1 = fractall gagne)", ""]
        if s.get("status") == "f3_unavailable":
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

    # Parity
    p = card.get("parity")
    if p and p.get("status") != "skipped":
        L += ["## Parité (compare_f3 — Δsmooth-iter vs F3)", ""]
        if p.get("status") == "f3_unavailable":
            L.append("_F3 indisponible — axe non exécuté._\n")
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
    L.append(f"_Scorecards versionnés : `harness/history/` · baseline : "
             f"`harness/baseline.json`. Généré par `scripts/harness.py`._")
    return "\n".join(L) + "\n"

def _fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"

def latest_history() -> Path | None:
    files = sorted(HISTORY_DIR.glob("*.json"))
    return files[-1] if files else None

def write_history(card: dict) -> Path:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    m = card["meta"]
    path = HISTORY_DIR / f"{m['stamp']}-{m['git_sha']}.json"
    path.write_text(json.dumps(card, indent=2, ensure_ascii=False))
    return path

def cmd_score(args) -> None:
    cfg = tier_config(args.tier)
    cases = ([c.strip() for c in args.cases.split(",")] if args.cases
             else cfg["cases"])
    width = args.width or cfg["width"]
    height = args.height or cfg["height"]
    runs = args.runs or cfg["runs"]
    timeout = args.timeout or cfg["timeout"]
    axes = [a.strip() for a in args.axes.split(",") if a.strip()]

    # Garde-fou crash : un breadcrumb résiduel = le run précédent est mort
    # pendant un cas (OOM / plantage OS). On le journalise + quarantaine AVANT
    # de relancer, sinon on replante sur le même cas.
    check_stale_inflight(auto_quarantine=not args.no_quarantine)
    if not args.no_quarantine:
        reconcile_quarantine_from_journal()
    mem_limit_mb = (0 if args.no_mem_limit
                    else args.mem_limit_mb if args.mem_limit_mb is not None
                    else default_mem_limit_mb())
    quarantined = set() if args.no_quarantine else set(load_quarantine())
    slow_safe = set(load_slow_safe())
    if mem_limit_mb:
        print(f"→ cap mémoire/process : {mem_limit_mb} MB "
              f"(RLIMIT_AS ; --no-mem-limit pour désactiver)")
    if quarantined:
        print(f"→ quarantaine active ({len(quarantined)}) : "
              f"{', '.join(sorted(quarantined))}")
    if slow_safe:
        print(f"→ slow-safe (skip, correct hors budget temps) ({len(slow_safe)}) : "
              f"{', '.join(sorted(slow_safe))}")

    if not args.no_rebuild:
        print("→ cargo build --release", flush=True)
        if not cargo_build():
            sys.exit("cargo build a échoué")

    f3_bin = compare_f3.find_f3()
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")

    card: dict = {
        "schema": SCHEMA,
        "meta": {
            "date_utc": now.replace(microsecond=0).isoformat(),
            "stamp": stamp,
            "git_sha": git_sha(),
            "git_dirty": git_dirty(),
            "machine": machine_info(),
            "tier": args.tier,
            "axes": axes,
            "width": width,
            "height": height,
            "quality_width": cfg.get("quality_width", 256),
            "quality_height": cfg.get("quality_height", 256),
            "runs": runs,
            "f3_bin": str(f3_bin) if f3_bin else None,
            "mem_limit_mb": mem_limit_mb,
            "quarantined": sorted(quarantined),
            "slow_safe": sorted(slow_safe),
        },
        "speed": {"status": "skipped"},
        "parity": {"status": "skipped"},
        "quality": {"status": "skipped"},
        "fuzz": {"status": "skipped"},
        "goldens": {"status": "skipped"},
    }

    print(f"→ tier={args.tier} cases={len(cases)} {width}×{height} "
          f"runs={runs} timeout={timeout}s axes={axes}")
    print(f"→ F3: {f3_bin or 'INDISPONIBLE (speed/parity dégradés)'}")

    if "speed" in axes:
        print("\n== SPEED ==")
        card["speed"] = axis_speed(cases, width, height, runs, timeout, f3_bin,
                                   mem_limit_mb=mem_limit_mb,
                                   quarantined=quarantined, slow_safe=slow_safe)
    if "parity" in axes:
        print("\n== PARITY ==")
        card["parity"] = axis_parity(cases, width, height, timeout, f3_bin,
                                     mem_limit_mb=mem_limit_mb)
    if "quality" in axes:
        print("\n== QUALITY ==")
        card["quality"] = axis_quality(
            args.no_rebuild,
            cfg.get("quality_width", 256),
            cfg.get("quality_height", 256),
            mem_limit_mb=mem_limit_mb,
        )
    if "fuzz" in axes:
        print("\n== FUZZ ==")
        card["meta"]["fuzz_seed"] = args.fuzz_seed
        card["fuzz"] = axis_fuzz(cfg.get("fuzz_probes", 3), args.fuzz_seed,
                                 mem_limit_mb=mem_limit_mb)
    if "goldens" in axes:
        print("\n== GOLDENS ==")
        card["goldens"] = axis_goldens()

    card["beyond_f3"] = {
        "speed_wins": card.get("speed", {}).get("wins", []),
        "correctness_wins": [
            n for n, c in card.get("parity", {}).get("cases", {}).items()
            if c.get("status") == "f3_degenerate"],
    }

    base = None
    if BASELINE.exists():
        try:
            base = json.loads(BASELINE.read_text())
        except Exception:
            base = None
    card["gaps"] = compute_gaps(card, base)

    hist = write_history(card)
    SCORECARD.write_text(build_scorecard_md(card, base))
    print(f"\n✓ history : {hist.relative_to(REPO)}")
    print(f"✓ scorecard : {SCORECARD.relative_to(REPO)}")
    _print_gaps(card["gaps"])

def cmd_baseline(_args) -> None:
    latest = latest_history()
    if latest is None:
        sys.exit("Aucun history — lance d'abord `harness.py score`.")
    shutil.copyfile(latest, BASELINE)
    print(f"✓ baseline ← {latest.relative_to(REPO)}")

def cmd_gaps(_args) -> None:
    latest = latest_history()
    if latest is None:
        sys.exit("Aucun history — lance d'abord `harness.py score`.")
    card = json.loads(latest.read_text())
    print(f"Gaps du dernier score ({latest.name}) :\n")
    _print_gaps(card.get("gaps", []))

def _print_gaps(gaps: list[dict]) -> None:
    gaps = sort_gaps(gaps)
    if not gaps:
        print("Aucun gap détecté 🎉")
        return
    print(f"\n{len(gaps)} gap(s) — top 10 :")
    for i, g in enumerate(gaps[:10], 1):
        sev = SEVERITY_LABEL[g["severity"]]
        print(f"  {i:>2}. [{g['severity']} {sev:<10}] {g['axis']}/{g['case']} "
              f"{g['metric']}={g['value']} — {g['note']}")

# --- preflight : vet le corpus SANS faire tomber l'OS -------------------------

_TIME_BIN = Path("/usr/bin/time")


def run_case_measured(name: str, width: int, height: int, timeout: float,
                      mem_limit_mb: int) -> dict:
    """Rend UN cas sous cap mémoire + breadcrumb, mesure temps et pic RSS.

    Renvoie {status, secs, peak_rss_kb, signal?}. status : ok | timeout |
    fail | killed_oom | aborted | killed | missing_toml. Le pic RSS vient de
    `/usr/bin/time -v` si dispo (mesure propre par process).
    """
    toml_path = TOML_DIR / f"{name}.toml"
    if not toml_path.exists():
        return {"status": "missing_toml"}
    out = BENCH / "preflight"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [str(CLI), "--toml", str(toml_path), "--width", str(width),
           "--height", str(height), "--bailout", "25",
           "--output", str(out / f"{name}.png")]
    use_time = _TIME_BIN.exists()
    full = (["/usr/bin/time", "-v", *cmd] if use_time else cmd)
    env = os.environ.copy()
    env["FRACTALL_NO_AUTO_ADJUST"] = "1"
    env["FRACTALL_NO_PERIOD"] = "1"
    jrec = journal_begin("preflight", name, {"size": f"{width}x{height}",
                                             "toml": str(toml_path)})
    t0 = time.monotonic()
    try:
        proc = subprocess.run(full, stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE, env=env, timeout=timeout,
                              preexec_fn=_rlimit_preexec(mem_limit_mb),
                              text=True)
    except subprocess.TimeoutExpired:
        journal_end(jrec, "timeout")
        return {"status": "timeout", "secs": None, "peak_rss_kb": None}
    except OSError:
        journal_end(jrec, "fail")
        return {"status": "fail", "secs": None, "peak_rss_kb": None}
    secs = time.monotonic() - t0
    rc = proc.returncode
    peak_kb = None
    signal_num = None
    if use_time and proc.stderr:
        m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)",
                      proc.stderr)
        if m:
            peak_kb = int(m.group(1))
        # /usr/bin/time masque une mort par signal : il n'y a pas de rc<0, time
        # loggue "Command terminated by signal N" et sort avec 128+N. Sans ça,
        # un abort (SIGABRT 6 = alloc échouée sous cap) passerait pour un simple
        # `fail` — exactement ce qui a masqué seahorse/orion/opus2.
        ms = re.search(r"Command terminated by signal (\d+)", proc.stderr)
        if ms:
            signal_num = int(ms.group(1))
    if signal_num is None and rc < 0:
        signal_num = -rc
    if signal_num is None and use_time and rc >= 128:
        signal_num = rc - 128     # convention shell/time pour mort par signal
    status = _classify_rc(-signal_num) if signal_num else _classify_rc(rc)
    res = {"status": status, "secs": round(secs, 3), "peak_rss_kb": peak_kb}
    if signal_num:
        res["signal"] = signal_num
    journal_end(jrec, status, {"secs": res["secs"], "peak_rss_kb": peak_kb,
                               "signal": signal_num})
    return res


def cmd_preflight(args) -> None:
    """Passe le corpus un-par-un, SOUS cap mémoire, pour trouver le cas qui
    fait tomber l'OS AVANT le sweep — sans risque (un runaway est tué+loggé)."""
    check_stale_inflight(auto_quarantine=not args.no_quarantine)
    if not args.no_quarantine:
        reconcile_quarantine_from_journal()
    if args.cases:
        cases = [c.strip() for c in args.cases.split(",")]
    elif args.tier:
        cases = tier_config(args.tier)["cases"]
    else:
        cases = all_toml_stems()
    width = args.width or 256
    height = args.height or 256
    timeout = args.timeout or 300.0
    mem_limit_mb = (0 if args.no_mem_limit
                    else args.mem_limit_mb if args.mem_limit_mb is not None
                    else default_mem_limit_mb())
    flag_mb = args.flag_rss_mb if args.flag_rss_mb is not None else \
        int(default_mem_limit_mb() * 0.6) or None

    if not args.no_rebuild and not CLI.exists():
        cargo_build("fractall-cli")

    print(f"→ preflight {len(cases)} cas · {width}×{height} · "
          f"timeout={timeout}s · cap={mem_limit_mb or 'off'}MB · "
          f"flag>{flag_mb or '—'}MB")
    slow_safe = set(load_slow_safe())
    rows: list[dict] = []
    for i, name in enumerate(cases, 1):
        print(f"  [{i:>3}/{len(cases)}] {name:28s}", end=" ", flush=True)
        if name in slow_safe:
            # correct + memory-safe attesté, hors budget temps → ne pas le
            # rejouer (il timeout → re-quarantaine à tort). Skip explicite.
            rows.append({"name": name, "status": "slow_safe",
                         "secs": None, "peak_rss_kb": None})
            print("slow_safe   (skip — voir slow-safe.json)")
            continue
        r = run_case_measured(name, width, height, timeout, mem_limit_mb)
        r["name"] = name
        rows.append(r)
        rss_mb = (r.get("peak_rss_kb") or 0) / 1024
        st = r["status"]
        # hog = quel que soit ok/fail : un cas à ~28 GB qui ÉCHOUE (alloc
        # refusée par le cap) est encore pire qu'un cas à 28 GB qui réussit —
        # les deux doivent sortir du sweep (bug initial : `and st==ok`).
        hog = bool(flag_mb) and rss_mb >= flag_mb
        crashed = st in ("killed_oom", "aborted", "killed", "timeout")
        note = ""
        if crashed or hog:
            bits = []
            if hog:
                bits.append(f"pic RSS {rss_mb:.0f}MB ≥ seuil {flag_mb}MB")
            if crashed or st != "ok":
                bits.append(st)
            reason = " ; ".join(bits) or st
            if not args.no_quarantine and add_quarantine(name, reason):
                note = "  → QUARANTAINE"
        print(f"{st:11s} {r.get('secs', '—')}s "
              f"rss={rss_mb:.0f}MB{note}")

    # rapport trié par pic RSS puis temps
    def _key(r):
        return (-(r.get("peak_rss_kb") or 0), -(r.get("secs") or 0))
    rows.sort(key=_key)
    out = BENCH / "preflight"
    out.mkdir(parents=True, exist_ok=True)
    report = out / "preflight-report.md"
    L = [f"# Preflight — {_now_iso()}  (`{git_sha()}`)", "",
         f"- corpus : {len(cases)} cas · {width}×{height} · "
         f"cap {mem_limit_mb or 'off'}MB · flag ≥ {flag_mb or '—'}MB", "",
         "| Cas | Statut | Temps (s) | Pic RSS (MB) | Signal |",
         "|---|---|---:|---:|---:|"]
    for r in rows:
        rss = f"{(r.get('peak_rss_kb') or 0)/1024:.0f}" if r.get("peak_rss_kb") \
            else "—"
        L.append(f"| {r['name']} | {r['status']} | {r.get('secs', '—')} | "
                 f"{rss} | {r.get('signal', '')} |")
    report.write_text("\n".join(L) + "\n")
    q = load_quarantine()
    print(f"\n✓ rapport : {report.relative_to(REPO)}")
    print(f"✓ quarantaine : {len(q)} cas — {', '.join(sorted(q)) or '(aucun)'}")
    print(f"✓ journal incidents : {CRASH_JOURNAL.relative_to(REPO)}")


def cmd_quarantine(args) -> None:
    action = args.action
    if action == "list":
        q = load_quarantine()
        if not q:
            print("Quarantaine vide.")
        else:
            print(f"{len(q)} cas en quarantaine :")
            for name, info in sorted(q.items()):
                print(f"  - {name}  ({info.get('added_utc', '?')}) "
                      f": {info.get('reason', '')}")
        res = load_resolved()
        if res:
            print(f"\n{len(res)} tombstone(s) résolu(s) (fix attesté, non "
                  f"re-quarantainé sauf nouvel incident) :")
            for name, ts in sorted(res.items()):
                print(f"  - {name}  (résolu {ts})")
        ss = load_slow_safe()
        if ss:
            print(f"\n{len(ss)} cas slow-safe (correct + memory-safe, hors "
                  f"budget temps — skip sweeps, PAS un gap) :")
            for name, info in sorted(ss.items()):
                print(f"  - {name}  ({info.get('added_utc', '?')}) "
                      f": {info.get('note', '')}")
    elif action == "add":
        if not args.case:
            sys.exit("quarantine add <case> [--reason ...]")
        ok = add_quarantine(args.case, args.reason)
        print(f"{'+ ajouté' if ok else '= déjà présent'} : {args.case}")
    elif action == "remove":
        if not args.case:
            sys.exit("quarantine remove <case>")
        ok = remove_quarantine(args.case)
        print(f"{'- retiré' if ok else '? absent'} : {args.case}")
    elif action == "clear":
        save_quarantine({})
        print("Quarantaine vidée.")


def cmd_slow_safe(args) -> None:
    """Registre slow-safe : cas CORRECTS et memory-safe mais hors budget temps
    des sweeps (deep-zoom extrême). Attesté par l'opérateur APRÈS mesure (rendu
    complet exit 0 + pic RSS < cap) — cf. SLOW_SAFE. `add` retire aussi le cas
    de la quarantaine (un cas prouvé sûr n'est plus un crash-danger)."""
    action = args.action
    d = load_slow_safe()
    if action == "list":
        if not d:
            print("Registre slow-safe vide.")
            return
        print(f"{len(d)} cas slow-safe :")
        for name, info in sorted(d.items()):
            print(f"  - {name}  ({info.get('added_utc', '?')}) "
                  f": {info.get('note', '')}")
    elif action == "add":
        if not args.case:
            sys.exit("slow-safe add <case> [--note ...]")
        d[args.case] = {"added_utc": _now_iso(),
                        "note": args.note or "correct + memory-safe, hors "
                                             "budget temps sweep"}
        save_slow_safe(d)
        # un cas prouvé sûr sort de la quarantaine crash/OOM (déconflation)
        if remove_quarantine(args.case):
            print(f"+ {args.case} slow-safe ; retiré de la quarantaine")
        else:
            print(f"+ {args.case} slow-safe")
    elif action == "remove":
        if not args.case:
            sys.exit("slow-safe remove <case>")
        if args.case in d:
            del d[args.case]
            save_slow_safe(d)
            print(f"- retiré : {args.case}")
        else:
            print(f"? absent : {args.case}")


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
        n_pass += verdict == "PASS"
        n_warn += verdict == "WARN"
        n_fail += verdict == "FAIL"
        n_adj += verdict == "ADJUDICATE"
        rows.append({"case": name, "status": "ok", "verdict": verdict,
                     "ratio_chosen_vs_best": round(ratio, 3), "noisy": noisy,
                     "output_differs": differs,
                     "chosen_s": round(chosen, 4), "best_alt": best_var,
                     "best_alt_s": round(best_alt, 4),
                     "routed_path": routed, "times": times})
        tag = " noisy" if noisy else " sortie≠" if differs else ""
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
        L.append(f"| {r['case']} | {r['verdict']} | {r['routed_path']} | "
                 f"{r['ratio_chosen_vs_best']} | {r['chosen_s']*1000:.0f} | "
                 f"{r['best_alt']} {r['best_alt_s']*1000:.0f}ms | {d} |")
    (out / "wisdom-opt.md").write_text("\n".join(L) + "\n")
    print(f"\n✓ rapport : {(out / 'wisdom-opt.md').relative_to(REPO)}")
    print(f"→ **{n_pass} PASS · {n_warn} WARN · {n_fail} FAIL · "
          f"{n_adj} ADJUDICATE**")
    if n_fail:
        print("⚠ wisdom SUB-OPTIMAL (plus lent à sortie IDENTIQUE) — seuil "
              "routage à recalibrer, voir le rapport.")
    if n_adj:
        print("• ADJUDICATE : alternative plus rapide mais sortie ≠ — vérifier "
              "correction vs GMP avant toute recalibration.")


def cmd_journal(_args) -> None:
    if not CRASH_JOURNAL.exists():
        print("Aucun incident journalisé 🎉")
        return
    lines = CRASH_JOURNAL.read_text().splitlines()
    print(f"{len(lines)} incident(s) — {CRASH_JOURNAL.relative_to(REPO)} :\n")
    for ln in lines[-30:]:
        try:
            r = json.loads(ln)
        except Exception:
            continue
        print(f"  [{r.get('ended_utc') or r.get('detected_utc', '?')}] "
              f"{r.get('outcome', '?'):15s} {r.get('phase', '?')}/"
              f"{r.get('case', '?')} "
              f"{('sig=' + str(r['signal'])) if r.get('signal') else ''} "
              f"{r.get('note', '')}")

def cmd_adjudicate(args) -> None:
    """Arbitre 3-voies (fractall-pert / F3 / GMP per-pixel) d'un cas parité,
    et PERSISTE le verdict dans harness/adjudications.json (versionné). Les
    scores suivants annotent les gaps parité avec ce verdict — un cas adjugé
    `f3_wrong` est déclassé (sévérité 2→4) : F3 fautif ≠ gap moteur fractall.
    ⚠️ coût : GMP per-pixel ≈ 1 µs/iter × size² — garder --size petit sur les
    cas profonds (>100k iters)."""
    sys.path.insert(0, str(REPO / "scripts"))
    import three_way_gmp as tw
    if tw.c.F3 is None:
        sys.exit("F3 binaire introuvable — bash scripts/build_f3_linux.sh")
    outdir = BENCH / "three_way"
    outdir.mkdir(parents=True, exist_ok=True)
    adjudications = load_adjudications()
    for stem in args.cases:
        toml = TOML_DIR / f"{stem}.toml"
        if not toml.exists():
            print(f"  {stem}: toml absent — skip")
            continue
        t = compare_f3.parse_light_toml(toml)
        iters = int(t.iterations or 1000)
        print(f"→ adjudication {stem} ({args.size}² · {iters} iters · "
              f"zoom {t.zoom})", flush=True)
        jrec = journal_begin("adjudicate", stem, {"size": f"{args.size}"})
        try:
            _, _, fp, f3s, total = tw.compare(
                stem, t.real, t.imag, t.zoom, iters, outdir,
                args.size, args.size)
        except Exception as e:  # rendu F3/fractall/GMP échoué
            journal_end(jrec, "fail", {"note": str(e)})
            print(f"  {stem}: ÉCHEC ({e})")
            continue
        journal_end(jrec, "ok")
        # Verdict sur les erreurs LARGES (>5, débruité du ±1 cross-implé).
        fb, f3b = fp["big"], f3s["big"]
        if fb <= max(2, total // 5000) and f3b > 4 * max(fb, 1):
            verdict = "f3_wrong"
        elif f3b <= max(2, total // 5000) and fb > 4 * max(f3b, 1):
            verdict = "fractall_wrong"
        elif fb > 0 and f3b > 0:
            verdict = "shared"
        else:
            verdict = "both_match_gmp"
        adjudications[stem] = {
            "date": _now_iso(), "git_sha": git_sha(), "size": args.size,
            "iters": iters, "fractall_big": fb, "f3_big": f3b,
            "fractall_max": fp["max"], "f3_max": f3s["max"],
            "verdict": verdict,
        }
        print(f"  {stem}: fractall big={fb} (max {fp['max']}) · "
              f"F3 big={f3b} (max {f3s['max']}) → {verdict}")
    ADJUDICATIONS.write_text(json.dumps(adjudications, indent=1,
                                        ensure_ascii=False) + "\n")
    print(f"\n✓ {ADJUDICATIONS.relative_to(REPO)} mis à jour "
          f"({len(adjudications)} cas adjugés)")

def main() -> None:
    install_signal_handlers()  # Ctrl-C/SIGTERM → pas de fausse quarantaine
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    sc = sub.add_parser("score", help="Mesure et écrit un scorecard.")
    sc.add_argument("--tier", choices=["quick", "standard", "full"],
                    default="quick")
    sc.add_argument("--axes", default="speed,parity,quality,fuzz,goldens")
    sc.add_argument("--fuzz-seed", type=int, default=FUZZ_SEED_DEFAULT,
                    help="seed des sondes fuzz (défaut committé ; le faire "
                         "tourner au rebaseline)")
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

    bl = sub.add_parser("baseline", help="Fige le dernier history comme baseline.")
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

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
