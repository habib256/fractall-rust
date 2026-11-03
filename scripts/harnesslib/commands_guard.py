"""Sous-commandes du garde-fou : preflight, quarantine, slow-safe."""

from __future__ import annotations

import sys

from .config import (
    BENCH, CLI, CRASH_JOURNAL, REPO, _now_iso, all_toml_stems, cargo_build,
    git_sha, tier_config,
)
from .guard import (
    add_quarantine, check_stale_inflight, default_mem_limit_mb,
    load_quarantine, load_resolved, load_slow_safe,
    reconcile_quarantine_from_journal, remove_quarantine, run_case_measured,
    save_quarantine, save_slow_safe,
)

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
