"""Sous-commandes d'INTROSPECTION : `doctor` (prérequis machine, axes
mesurables, état de la baseline) et `trend` (évolution des métriques dans
harness/history). Aucune des deux ne rend quoi que ce soit."""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path

from .config import (
    CLI, CRASH_JOURNAL, HISTORY_DIR, INFLIGHT, QUALITY, REPO,
    baseline_path_for, f3_unlaunchable, find_f3, git_dirty, git_sha, load_baseline, machine_info,
    mem_total_mb,
)
from .gaps import _same_machine
from .guard import (
    default_mem_limit_mb, load_quarantine, load_slow_safe, mem_cap_supported,
)
from .scorecard import _fmt, load_history, trend_table

OK, WARN, FAIL, INFO = "✅", "⚠️ ", "❌", "·"


def _importable(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:  # ImportError, SystemExit d'un module récalcitrant…
        return False


def _wisdom_file() -> Path | None:
    env = os.environ.get("FRACTALL_WISDOM_FILE")
    if env:
        return Path(env)
    home = os.environ.get("HOME")
    return Path(home) / ".config/fractall/wisdom.toml" if home else None


def _mem_total_gb() -> float | None:
    mb = mem_total_mb()
    return round(mb / 1024, 1) if mb else None


def doctor_report() -> dict:
    """Inventaire pur (pas d'affichage) : {checks: [(statut, label, détail)],
    axes: {axe: bool}, recommended: [commandes]}."""
    checks: list[tuple[str, str, str]] = []
    mi = machine_info()
    checks.append((INFO, "machine", f"{mi['cpu']} · {mi['nproc']} threads · "
                   f"{_mem_total_gb() or '?'} GB RAM · {mi['os']} · cap mémoire "
                   + (f"défaut {default_mem_limit_mb()} MB"
                      if mem_cap_supported()
                      else "non applicable (RLIMIT_AS Linux seulement)")))
    checks.append((INFO, "git", f"{git_sha()}" + (" (dirty)" if git_dirty() else "")))
    py_ok = sys.version_info >= (3, 11)
    checks.append((OK if py_ok else WARN, "python",
                   f"{sys.version.split()[0]} (tomllib requis ≥ 3.11 pour l'axe bench)"))
    cargo = shutil.which("cargo")
    checks.append((OK if cargo else FAIL, "cargo", cargo or "absent — rustup.rs"))
    for label, path in (("fractall-cli", CLI), ("fractall-quality", QUALITY)):
        checks.append((OK if path.exists() else WARN, label,
                       str(path.relative_to(REPO)) if path.exists()
                       else "non bâti — `cargo build --release` (score le fait sauf --no-rebuild)"))
    f3 = find_f3()
    broken = None if f3 else f3_unlaunchable()
    if broken:
        f3_detail = (f"{broken[0].name} présent mais NON lançable — "
                     f"{len(broken[1])} dylib(s) absente(s) (ex. "
                     f"{Path(broken[1][0]).name}) → axes speed(ratio)/parity "
                     "éteints ; `brew install openexr imath` ou rebâtir F3")
    else:
        f3_detail = ("absent → axes speed(ratio)/parity éteints ; "
                     "build : bash scripts/build_f3_linux.sh (ou F3_BIN=…)")
    checks.append((OK if f3 else WARN, "Fraktaler-3",
                   str(f3) if f3 else f3_detail))
    deps = {m: _importable(m) for m in ("numpy", "OpenEXR", "PIL")}
    missing = [m for m, ok in deps.items() if not ok]
    checks.append((OK if not missing else WARN, "deps Python EXR (parité)",
                   "numpy, OpenEXR, Pillow OK" if not missing
                   else f"manquent : {', '.join(missing)} → "
                        f"pip3 install numpy OpenEXR Pillow (axe parity + adjudicate)"))
    checks.append((OK if _importable("mpmath") else INFO, "mpmath",
                   "OK (independent_probe.py)" if _importable("mpmath")
                   else "absent — optionnel (scripts/independent_probe.py)"))
    tb = Path("/usr/bin/time")
    checks.append((OK if tb.exists() else WARN, "/usr/bin/time",
                   "OK (pic RSS du preflight)" if tb.exists()
                   else "absent → preflight sans pic RSS (apt install time)"))
    ff = shutil.which("ffmpeg")
    checks.append((OK if ff else INFO, "ffmpeg", ff or "absent — optionnel (fractall-video)"))
    wf = _wisdom_file()
    checks.append((OK if (wf and wf.exists()) else INFO, "wisdom machine",
                   f"{wf}" if (wf and wf.exists())
                   else f"absent ({wf}) — `fractall-cli --wisdom-bench` pour "
                        f"l'arbitrage device G9.5 (l'axe bench du harness est isolé)"))
    # baseline
    base = load_baseline(mi)
    if base:
        bm = base["meta"]
        same = _same_machine({"meta": {"machine": mi}}, base)
        checks.append((OK if same else WARN, "baseline",
                       f"{base.get('_file')} · {bm.get('git_sha')} · tier "
                       f"{bm.get('tier')} · {bm.get('date_utc', '?')[:10]} · machine "
                       f"{'IDENTIQUE' if same else 'DIFFÉRENTE (' + str(bm.get('machine', {}).get('cpu')) + ') → pas de delta vitesse ; `harness.py baseline` après un premier score ici écrira ' + str(baseline_path_for(mi).relative_to(REPO))}"))
    else:
        checks.append((WARN, "baseline", f"absente — `harness.py score` puis "
                       f"`harness.py baseline` → {baseline_path_for(mi).relative_to(REPO)}"))
    n_hist = len(list(HISTORY_DIR.glob("*.json"))) if HISTORY_DIR.exists() else 0
    n_here = len(load_history(same_machine_as={"meta": {"machine": mi}}))
    checks.append((INFO, "history", f"{n_hist} scorecards ({n_here} sur cette machine)"))
    # garde-fou
    if INFLIGHT.exists():
        checks.append((FAIL, "inflight", "breadcrumb résiduel = crash du run précédent "
                       "(score/preflight le journalisent + quarantainent)"))
    else:
        checks.append((OK, "inflight", "aucun crash en suspens"))
    n_inc = (len(CRASH_JOURNAL.read_text().splitlines()) if CRASH_JOURNAL.exists() else 0)
    q, ss = load_quarantine(), load_slow_safe()
    checks.append((INFO, "journal/quarantaine",
                   f"{n_inc} incident(s) · {len(q)} en quarantaine "
                   f"{sorted(q) if q else ''} · {len(ss)} slow-safe {sorted(ss) if ss else ''}"))
    axes = {
        "speed (ratio F3)": bool(f3) and not missing,
        "speed (intrinsèque)": CLI.exists() or bool(cargo),
        "parity": bool(f3) and not missing,
        "quality": QUALITY.exists() or bool(cargo),
        "fuzz": QUALITY.exists() or bool(cargo),
        "goldens": bool(cargo),
        "profile": CLI.exists() or bool(cargo),
        "bench": (CLI.exists() or bool(cargo)) and py_ok,
    }
    rec = ["python3 scripts/harness.py score --tier quick"]
    if not f3:
        rec.append("bash scripts/build_f3_linux.sh      # rallume speed-ratio + parity")
    if missing:
        rec.append("pip3 install numpy OpenEXR Pillow   # parité + adjudicate")
    if not base or not _same_machine({"meta": {"machine": mi}}, base):
        rec.append("python3 scripts/harness.py baseline  # après le 1er score sur CETTE machine")
    return {"checks": checks, "axes": axes, "recommended": rec,
            "has_fail": any(c[0] == FAIL for c in checks)}


def cmd_doctor(args) -> None:
    rep = doctor_report()
    print("harness doctor — prérequis de la boucle d'auto-amélioration\n")
    for st, label, detail in rep["checks"]:
        print(f"  {st} {label:28s} {detail}")
    print("\nAxes mesurables ici :")
    for axe, ok in rep["axes"].items():
        print(f"  {'✅' if ok else '⊘ '} {axe}")
    print("\nCommandes recommandées :")
    for r in rep["recommended"]:
        print(f"  $ {r}")
    if args.strict and rep["has_fail"]:
        sys.exit(1)


def cmd_trend(args) -> None:
    me = {"meta": {"machine": machine_info()}}
    cards = load_history(same_machine_as=None if args.all_machines else me,
                         tier=args.tier, limit=args.last)
    if not cards:
        sys.exit("Aucun history correspondant (essayer --all-machines).")
    scope = "toutes machines" if args.all_machines else "cette machine"
    L = [f"# Tendance harness — {scope}, {len(cards)} score(s)"
         + (f", tier {args.tier}" if args.tier else ""), ""]
    if args.case:
        n = args.case
        L.append(f"Cas `{n}` : temps fractall / F3 / ratio / boucle pixel")
        L.append("")
        L.append("| Date (UTC) | Commit | Tier | fractall (s) | F3 (s) | ratio | "
                 "ns/iter | path | orbite % | pixels % |")
        L.append("|---|---|---|---:|---:|---:|---:|---|---:|---:|")
        for c in cards:
            m = c["meta"]
            s = (c.get("speed", {}).get("cases", {}) or {}).get(n) or {}
            pcase = (c.get("profile", {}).get("cases", {}) or {}).get(n) or {}
            sh = pcase.get("share_pct") or {}
            if not s and not pcase:
                continue
            L.append(f"| {m.get('date_utc', '?')[:16].replace('T', ' ')} | "
                     f"`{m.get('git_sha')}` | {m.get('tier')} | "
                     f"{_fmt(s.get('fractall_s'))} | {_fmt(s.get('f3_s'))} | "
                     f"{_fmt(s.get('ratio'))} | {_fmt(pcase.get('ns_per_iter'))} | "
                     f"{pcase.get('path') or '—'} | {sh.get('orbit', '—')} | "
                     f"{sh.get('pixels', '—')} |")
    elif args.bench:
        keys: list[str] = []
        for c in cards:
            for k in (c.get("bench", {}).get("entries", {}) or {}):
                if k not in keys:
                    keys.append(k)
        if not keys:
            sys.exit("Aucun axe bench dans l'historique sélectionné.")
        L.append("Débits `--wisdom-bench` (iters/s) par technique")
        L.append("")
        L.append("| Date (UTC) | Commit | " + " | ".join(keys) + " |")
        L.append("|---|---|" + "---:|" * len(keys))
        for c in cards:
            e = c.get("bench", {}).get("entries", {}) or {}
            if not e:
                continue
            m = c["meta"]
            cells = [f"{e[k]['iters_per_sec']:.3e}" if k in e else "—" for k in keys]
            L.append(f"| {m.get('date_utc', '?')[:16].replace('T', ' ')} | "
                     f"`{m.get('git_sha')}` | " + " | ".join(cells) + " |")
    else:
        L += trend_table(cards)
    text = "\n".join(L) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"✓ écrit : {args.out}")
    else:
        print(text)
