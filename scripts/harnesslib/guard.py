"""Garde-fou crash/mémoire : breadcrumb inflight, journal, quarantaine, slow-safe,
cap RLIMIT_AS, chronométrage et mesure RSS d'un cas."""

from __future__ import annotations

from pathlib import Path
import json
import os
import re
import signal
import statistics
import subprocess
import time

from .config import (
    BENCH, CLI, CRASH_JOURNAL, F3_DEGENERATE, HARD_CRASH_OUTCOMES,
    HARNESS_DIR, INFLIGHT, QUARANTINE, REPO, RESOLVED, SLOW_SAFE, TOML_DIR,
    _now_iso, git_sha,
)

try:
    import resource  # POSIX only
except ImportError:  # pragma: no cover
    resource = None

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

def timed_runs(cmd: list[str], env: dict, timeout: float, runs: int,
               accept_nonzero: bool = False,
               mem_limit_mb: int = 0) -> tuple[str, float | None]:
    """Chronomètre `cmd` `runs` fois. Renvoie (status, médiane|None).
    Enveloppe de `timed_runs_detail` (API historique)."""
    st, det = timed_runs_detail(cmd, env, timeout, runs, accept_nonzero,
                                mem_limit_mb)
    return st, (det["median"] if det else None)


def timed_runs_detail(cmd: list[str], env: dict, timeout: float, runs: int,
                      accept_nonzero: bool = False,
                      mem_limit_mb: int = 0) -> tuple[str, dict | None]:
    """Chronomètre `cmd` `runs` fois. Renvoie (status, {median, min, max,
    spread, all}|None) — `spread` = (max−min)/median, le bruit de mesure
    (None si runs=1) : un delta vs baseline inférieur au spread n'est pas une
    régression.

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
    med = statistics.median(times)
    spread = (round((max(times) - min(times)) / med, 4)
              if len(times) > 1 and med > 0 else None)
    return "ok", {"median": med, "min": min(times), "max": max(times),
                  "spread": spread, "all": [round(x, 4) for x in times]}

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
