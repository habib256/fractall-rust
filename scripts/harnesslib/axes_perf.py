"""Axes de PERFORMANCE intrinsèque (sans F3) : profil par phase et débits.

- `axis_profile` : rend chaque cas UNE fois en capturant les diagnostics stderr
  du moteur (`[FRACTALL]`, `[PERTURB PERF]`, `[WISDOM]`, `[DD-ESCALATION]`,
  `[RELIABILITY]`, `[NUCLEUS]`) et la ligne stdout du CLI « Génération
  terminée en Xs (rendu: Ys, sauvegarde: Zs) » → OÙ VA LE TEMPS (orbite
  référence / boucle pixel / post / sauvegarde / autre) et QUEL path tourne.
  C'est l'étape « confirmer le path exécuté avant d'optimiser » (leçon G2)
  rendue systématique et versionnée dans le history.
- `axis_bench` : `fractall-cli --wisdom-bench` ISOLÉ (`FRACTALL_WISDOM_FILE`
  dans bench/, le wisdom machine de l'utilisateur n'est pas touché) → iters/s
  effectifs par technique (cpu_std_f64, cpu_perturb_{f64,exp,dd}, gpu_*) :
  le ns/iter de chaque tier, comparable d'un commit à l'autre SUR LA MÊME
  MACHINE (régression > 15 % = gap vitesse).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

from .config import BENCH, CLI, TOML_DIR, machine_info
from .guard import _classify_rc, _rlimit_preexec, journal_begin, journal_end

try:
    import tomllib  # 3.11+
except ImportError:  # pragma: no cover
    tomllib = None

# Phases rapportées par le moteur (secondes). `bla` = build de la table BLA
# (prewarm, chronométré à part depuis G14.4a — absent des sorties antérieures,
# où il tombait dans `other`). `other` = résidu du wall-clock (démarrage
# process, colorisation, compilation bytecode…).
PHASES = ("orbit", "bla", "pixels", "post", "save", "other")

_RE_FRACTALL = re.compile(
    r"\[FRACTALL\] type=(?P<type>\S+) path=(?P<path>\S+) prec=(?P<prec>\d+)b "
    r"iter_max=(?P<iter_max>\d+) avg_iter/px=(?P<avg>[\d.]+) "
    r"max_iter/px=(?P<max>\d+) ns/iter=(?P<ns>[\d.]+) "
    r"pixels=(?P<pixels_s>[\d.]+)s total=(?P<total_s>[\d.]+)s")
_RE_PERF = re.compile(
    r"\[PERTURB PERF\] \d+x\d+ pixels=\d+ zoom=\S+ small_image=\S+ "
    r"orbit=(?P<orbit>[\d.]+)s (?:bla=(?P<bla>[\d.]+)s )?"
    r"pixels=(?P<pixels>[\d.]+)s post=(?P<post>[\d.]+)s "
    r"total=(?P<total>[\d.]+)s avg_iter/px=(?P<avg>[\d.]+) max_iter/px=(?P<max>\d+) "
    r"ns/iter=(?P<ns>[\d.]+) glitched_initial=(?P<glitched>\d+) "
    r"corrections=(?P<corr>\d+) fallback_ratio=(?P<fallback>[\d.]+)")
_RE_WISDOM = re.compile(
    r"\[WISDOM\] type=(?P<type>\S+) device=(?P<device>\S+) algo=(?P<algo>\S+) "
    r"tier=(?P<tier>\S+) variants=(?P<variants>\S+)")
_RE_RELIAB = re.compile(
    r"\[RELIABILITY\] mode=(?P<mode>\S+) runaway=\S+ "
    r"flagged=(?P<flagged>\d+)/(?P<total>\d+) \((?P<pct>[\d.]+)%\)")
_RE_GEN = re.compile(
    r"Génération terminée en (?P<total>[\d.]+)s \(rendu: (?P<render>[\d.]+)s, "
    r"sauvegarde: (?P<save>[\d.]+)s\)")
_RE_NUCLEUS = re.compile(r"\[NUCLEUS\] found period=(?P<period>\d+)")


def parse_profile_output(text: str) -> dict:
    """Extrait le profil d'une sortie CLI (stdout+stderr mélangés). Pure (testée
    par scripts/test_harness_guard.py). Plusieurs `[PERTURB PERF]` (escalade
    dd = 2 rendus) → phases SOMMÉES (chaque rendu coûte du wall réel) ; le
    `[FRACTALL]` retenu est le DERNIER (path final)."""
    out: dict = {"path": None, "algo": None, "tier": None, "device": None,
                 "ns_per_iter": None, "avg_iter_px": None, "max_iter_px": None,
                 "orbit_s": None, "bla_s": None, "pixels_s": None, "post_s": None,
                 "variants": None,
                 "render_s": None, "save_s": None, "gen_total_s": None,
                 "dd_escalation": "[DD-ESCALATION]" in text,
                 "reliability_flagged_pct": None, "nucleus_period": None,
                 "perturb_fallback_ratio": None, "n_perf_lines": 0}
    fr = None
    for m in _RE_FRACTALL.finditer(text):
        fr = m
    if fr:
        out["path"] = fr["path"]
        out["ns_per_iter"] = float(fr["ns"])
        out["avg_iter_px"] = float(fr["avg"])
        out["max_iter_px"] = int(fr["max"])
    perf = list(_RE_PERF.finditer(text))
    if perf:
        out["n_perf_lines"] = len(perf)
        out["orbit_s"] = round(sum(float(m["orbit"]) for m in perf), 4)
        if any(m["bla"] is not None for m in perf):
            out["bla_s"] = round(sum(float(m["bla"] or 0.0) for m in perf), 4)
        out["pixels_s"] = round(sum(float(m["pixels"]) for m in perf), 4)
        out["post_s"] = round(sum(float(m["post"]) for m in perf), 4)
        out["perturb_fallback_ratio"] = float(perf[-1]["fallback"])
        if out["ns_per_iter"] is None:
            out["ns_per_iter"] = float(perf[-1]["ns"])
            out["avg_iter_px"] = float(perf[-1]["avg"])
            out["max_iter_px"] = int(perf[-1]["max"])
    w = None
    for m in _RE_WISDOM.finditer(text):
        w = m
    if w:
        # ⚠️ `tier=` du log [WISDOM] est le LABEL du path (`bytecode_f64`,
        # `bytecode_exp`…) ; `variants=` la partie STATIQUE du routage
        # (ex. `harmonic_lla` = candidat, la décision finale est au build BLA).
        out["algo"], out["tier"], out["device"] = w["algo"], w["tier"], w["device"]
        out["variants"] = w["variants"]
    r = None
    for m in _RE_RELIAB.finditer(text):
        r = m
    if r:
        out["reliability_flagged_pct"] = float(r["pct"])
    g = _RE_GEN.search(text)
    if g:
        out["gen_total_s"] = float(g["total"])
        out["render_s"] = float(g["render"])
        out["save_s"] = float(g["save"])
    n = _RE_NUCLEUS.search(text)
    if n:
        out["nucleus_period"] = int(n["period"])
    return out


def phase_breakdown(prof: dict, wall_s: float | None) -> dict:
    """Secondes par phase + part (%) du wall-clock + phase dominante.
    Sans `[PERTURB PERF]` (path f64 standard) : tout le rendu est `pixels`."""
    total = wall_s or prof.get("gen_total_s") or 0.0
    orbit = prof.get("orbit_s") or 0.0
    bla = prof.get("bla_s") or 0.0
    post = prof.get("post_s") or 0.0
    render = prof.get("render_s")
    if prof.get("pixels_s") is not None:
        pixels = prof["pixels_s"]
    else:
        pixels = render or 0.0
    save = prof.get("save_s") or 0.0
    other = max(0.0, total - orbit - bla - pixels - post - save)
    secs = {"orbit": orbit, "bla": bla, "pixels": pixels, "post": post,
            "save": save, "other": other}
    share = {k: (round(100.0 * v / total, 1) if total > 0 else None)
             for k, v in secs.items()}
    dominant = max(secs, key=lambda k: secs[k]) if total > 0 else None
    return {"secs": {k: round(v, 4) for k, v in secs.items()},
            "share_pct": share, "dominant": dominant}


# Wrapper Python : lance la commande puis imprime le pic RSS (ko) de SON enfant
# sur la dernière ligne de stderr — `ru_maxrss` de RUSAGE_CHILDREN est le max
# sur les enfants TERMINÉS du process appelant ; dans un wrapper frais il n'y
# en a qu'un → mesure exacte par cas, sans dépendre de `/usr/bin/time`.
RSS_WRAPPER = (
    "import resource,subprocess,sys;"
    "rc=subprocess.run(sys.argv[1:]).returncode;"
    "print('[HARNESS-RSS-KB]', resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,"
    " file=sys.stderr);"
    "sys.exit(rc if rc >= 0 else 128 - rc)"
)
_RE_RSS = re.compile(r"\[HARNESS-RSS-KB\] (\d+)")


def effective_threads(env: dict | None = None) -> int:
    """Threads rayon du moteur : `RAYON_NUM_THREADS` s'il est posé, sinon le
    nombre de CPU logiques (défaut rayon)."""
    raw = (env if env is not None else os.environ).get("RAYON_NUM_THREADS", "")
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return os.cpu_count() or 1


def precise_ns_per_iter(prof: dict, n_pixels: int, threads: int) -> None:
    """Le `ns/iter=` du moteur est imprimé à 0,1 ns près : sur les cas à gros
    sauts BLA il s'affiche « 0.0 ». Recalcul depuis `pixels_s` et le nombre
    d'itérations (avg_iter/px × pixels), puis normalisation PAR THREAD
    (`ns_per_iter_thread`) — la seule forme comparable d'une machine à l'autre.
    Sans phases mesurées (path std : pas de `[PERTURB PERF]`) : inchangé."""
    prof["threads"] = threads
    prof["ns_per_iter_thread"] = None
    px_s, avg = prof.get("pixels_s"), prof.get("avg_iter_px")
    if px_s is None or not avg or n_pixels <= 0:
        return
    ns = px_s * 1e9 / (avg * n_pixels)
    prof["ns_per_iter"] = round(ns, 4)
    prof["ns_per_iter_thread"] = round(ns * threads, 4)


def wrap_rss(cmd: list[str]) -> list[str]:
    return [sys.executable, "-I", "-c", RSS_WRAPPER, *cmd]


def parse_rss_kb(text: str) -> int | None:
    m = None
    for m in _RE_RSS.finditer(text):
        pass
    if m is None:
        return None
    kb = int(m.group(1))
    # macOS rapporte ru_maxrss en OCTETS, Linux en ko.
    return kb // 1024 if sys.platform == "darwin" else kb


def axis_profile(cases: list[str], width: int, height: int, timeout: float,
                 mem_limit_mb: int = 0, quarantined: set | None = None,
                 slow_safe: set | None = None) -> dict:
    outdir = BENCH / "profile"
    outdir.mkdir(parents=True, exist_ok=True)
    quarantined = quarantined or set()
    slow_safe = slow_safe or set()
    env = os.environ.copy()
    env.update({"FRACTALL_NO_AUTO_ADJUST": "1", "FRACTALL_NO_PERIOD": "1",
                "FRACTALL_PERTURB_STATS": "1", "FRACTALL_WISDOM": "1"})
    results: dict[str, dict] = {}
    for i, name in enumerate(cases, 1):
        print(f"  [{i:>3}/{len(cases)}] profile {name:22s}", end=" ", flush=True)
        if name in quarantined or name in slow_safe:
            results[name] = {"status": "quarantined" if name in quarantined
                             else "slow_safe"}
            print("⊘ skip")
            continue
        toml_path = TOML_DIR / f"{name}.toml"
        if not toml_path.exists():
            results[name] = {"status": "missing_toml"}
            print("✗ toml absent")
            continue
        cmd = [str(CLI), "--toml", str(toml_path), "--width", str(width),
               "--height", str(height), "--bailout", "25",
               "--output", str(outdir / f"{name}.png")]
        jrec = journal_begin("profile", name, {"size": f"{width}x{height}"})
        t0 = time.monotonic()
        try:
            proc = subprocess.run(wrap_rss(cmd), stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, env=env,
                                  timeout=timeout, text=True,
                                  errors="replace",
                                  preexec_fn=_rlimit_preexec(mem_limit_mb))
        except subprocess.TimeoutExpired:
            journal_end(jrec, "timeout")
            results[name] = {"status": "timeout"}
            print("⏱ timeout")
            continue
        wall = time.monotonic() - t0
        status = _classify_rc(proc.returncode)
        journal_end(jrec, status, {"secs": round(wall, 3)})
        (outdir / f"{name}.log").write_text(proc.stdout or "")
        if status != "ok":
            results[name] = {"status": status, "wall_s": round(wall, 4)}
            print(f"✗ {status}")
            continue
        prof = parse_profile_output(proc.stdout or "")
        prof["status"] = "ok"
        prof["wall_s"] = round(wall, 4)
        rss_kb = parse_rss_kb(proc.stdout or "")
        prof["peak_rss_mb"] = round(rss_kb / 1024, 1) if rss_kb else None
        precise_ns_per_iter(prof, width * height, effective_threads(env))
        prof.update(phase_breakdown(prof, wall))
        results[name] = prof
        sh = prof["share_pct"]
        print(f"{prof['path'] or '?':28s} wall={wall:.3f}s "
              f"orb={sh['orbit']}% bla={sh['bla']}% px={sh['pixels']}% post={sh['post']}% "
              f"save={sh['save']}% ns/it={prof['ns_per_iter']} "
              f"rss={prof['peak_rss_mb']}MB"
              f"{' DD-ESC' if prof['dd_escalation'] else ''}")
    ok = [r for r in results.values() if r.get("status") == "ok"]
    cum = {k: round(sum(r["secs"][k] for r in ok), 4) for k in PHASES}
    tot = sum(cum.values())
    cum_share = {k: (round(100.0 * v / tot, 1) if tot > 0 else None)
                 for k, v in cum.items()}
    dominant_counts: dict[str, int] = {}
    for r in ok:
        d = r.get("dominant")
        if d:
            dominant_counts[d] = dominant_counts.get(d, 0) + 1
    rss = [r["peak_rss_mb"] for r in ok if r.get("peak_rss_mb")]
    return {"status": "ok", "cases": results, "n_ok": len(ok),
            "threads": effective_threads(env),
            "peak_rss_mb_max": max(rss) if rss else None,
            "cumulative_secs": cum, "cumulative_share_pct": cum_share,
            "dominant_counts": dominant_counts,
            "dd_escalations": sorted(n for n, r in results.items()
                                     if r.get("dd_escalation")),
            "paths": sorted({r["path"] for r in ok if r.get("path")})}


_RE_BENCH_LINE = re.compile(
    r"\[WISDOM-BENCH\] (?P<key>\w+) : (?P<ips>[\d.e+\-]+) iters/s "
    r"\((?P<w>\d+)x(?P<h>\d+), (?P<secs>[\d.]+)s\)")


def parse_bench_toml(text: str) -> dict[str, dict]:
    """`wisdom.toml` → {key: {iters_per_sec, width, height, seconds}}."""
    entries: dict[str, dict] = {}
    if tomllib is not None:
        try:
            data = tomllib.loads(text)
        except Exception:
            data = {}
        for e in data.get("bench", []) or []:
            if isinstance(e, dict) and "key" in e:
                entries[str(e["key"])] = {
                    "iters_per_sec": float(e.get("iters_per_sec", 0.0)),
                    "width": int(e.get("width", 0)),
                    "height": int(e.get("height", 0)),
                    "seconds": float(e.get("seconds", 0.0))}
    return entries


def axis_bench(mem_limit_mb: int = 0, timeout: float = 900.0) -> dict:
    outdir = BENCH / "bench"
    outdir.mkdir(parents=True, exist_ok=True)
    wisdom_file = outdir / "wisdom.toml"
    if wisdom_file.exists():
        wisdom_file.unlink()
    env = os.environ.copy()
    env["FRACTALL_WISDOM_FILE"] = str(wisdom_file)   # isolé du wisdom utilisateur
    print("  $ fractall-cli --wisdom-bench  (FRACTALL_WISDOM_FILE=bench/…)",
          flush=True)
    jrec = journal_begin("bench", "<wisdom-bench>")
    t0 = time.monotonic()
    try:
        proc = subprocess.run([str(CLI), "--wisdom-bench"], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, errors="replace", timeout=timeout,
                              preexec_fn=_rlimit_preexec(mem_limit_mb))
    except subprocess.TimeoutExpired:
        journal_end(jrec, "timeout")
        return {"status": "timeout"}
    status = _classify_rc(proc.returncode)
    journal_end(jrec, status, {"secs": round(time.monotonic() - t0, 3)})
    (outdir / "wisdom-bench.log").write_text(proc.stdout or "")
    entries: dict[str, dict] = {}
    if wisdom_file.exists():
        entries = parse_bench_toml(wisdom_file.read_text())
    if not entries:  # repli : lignes stderr
        for m in _RE_BENCH_LINE.finditer(proc.stdout or ""):
            entries[m["key"]] = {"iters_per_sec": float(m["ips"]),
                                 "width": int(m["w"]), "height": int(m["h"]),
                                 "seconds": float(m["secs"])}
    if not entries:
        return {"status": "parse_error" if status == "ok" else status}
    for k, e in entries.items():
        ips = e["iters_per_sec"]
        e["ns_per_iter"] = round(1e9 / ips, 3) if ips > 0 else None
        print(f"    {k:18s} {ips:12.3e} iters/s  ({e['ns_per_iter']} ns/iter, "
              f"{e['width']}², {e['seconds']:.2f}s)")
    (outdir / "bench.json").write_text(json.dumps(
        {"machine": machine_info(), "entries": entries}, indent=1))
    return {"status": "ok", "entries": entries,
            "wall_s": round(time.monotonic() - t0, 3)}
