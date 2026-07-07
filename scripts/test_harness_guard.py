#!/usr/bin/env python3
"""Verrou du garde-fou crash : réconciliation journal↔quarantaine.

Régression couverte : un cas `died_uncleanly` dans le journal append-only qui
n'est PAS en quarantaine (quarantine.json dérivé/reverté) DOIT être re-exclu des
sweeps (invariant « crash journalisé ⇒ quarantainé »), sans réintégrer les cas
résolus manuellement ni les interruptions gracieuses.

Lancer : python3 scripts/test_harness_guard.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402


def _use_tmp(tmp: Path):
    harness.HARNESS_DIR = tmp
    harness.CRASH_JOURNAL = tmp / "crash-journal.jsonl"
    harness.QUARANTINE = tmp / "quarantine.json"
    harness.RESOLVED = tmp / "resolved.json"


def _journal(recs):
    with harness.CRASH_JOURNAL.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ✓ {msg}")


def run():
    with tempfile.TemporaryDirectory() as d:
        _use_tmp(Path(d))

        # 1. died_uncleanly loose → re-quarantainé
        _journal([
            {"case": "e22522", "phase": "speed_f3",
             "outcome": "died_uncleanly", "detected_utc": "2026-07-07T09:16:10+00:00"},
            {"case": "triangle", "phase": "preflight",
             "outcome": "interrupted", "ended_utc": "2026-07-07T09:28:27+00:00"},
            {"case": "somef3", "phase": "speed_f3",
             "outcome": "fail", "ended_utc": "2026-07-07T09:30:00+00:00"},
            {"case": "hog", "phase": "preflight",
             "outcome": "aborted", "detected_utc": "2026-07-07T08:10:20+00:00"},
        ])
        harness.save_quarantine({})  # dérive : quarantaine vidée
        readded = harness.reconcile_quarantine_from_journal()
        q = harness.load_quarantine()
        _check("e22522" in q, "died_uncleanly loose → re-quarantainé")
        _check("hog" in q, "aborted loose → re-quarantainé")
        _check("triangle" not in q, "interrupted gracieux → PAS quarantainé")
        _check("somef3" not in q, "fail bénin (rc≠0) → PAS quarantainé")
        _check(set(readded) == {"e22522", "hog"}, "readded == cas hard-crash")

        # 2. idempotent : re-lancer ne double rien
        readded2 = harness.reconcile_quarantine_from_journal()
        _check(readded2 == [], "réconciliation idempotente (rien à re-ajouter)")

        # 3. remove pose un tombstone → PAS re-quarantainé
        harness.remove_quarantine("e22522")
        _check("e22522" in harness.load_resolved(), "remove → tombstone résolu")
        readded3 = harness.reconcile_quarantine_from_journal()
        _check("e22522" not in readded3,
               "résolu après incident → non re-quarantainé")
        _check("e22522" not in harness.load_quarantine(),
               "résolu reste hors quarantaine")

        # 4. incident PLUS RÉCENT que le tombstone → re-quarantainé
        with harness.CRASH_JOURNAL.open("a") as f:
            f.write(json.dumps({"case": "e22522", "phase": "speed",
                                "outcome": "killed_oom",
                                "detected_utc": "2027-01-01T00:00:00+00:00"}) + "\n")
        readded4 = harness.reconcile_quarantine_from_journal()
        _check("e22522" in readded4,
               "nouvel incident postérieur au tombstone → re-quarantainé")

        # 5. add périme le tombstone résolu
        harness.save_quarantine({})
        harness.save_resolved({"hog": "2027-06-01T00:00:00+00:00"})
        harness.add_quarantine("hog", "manuel")
        _check("hog" not in harness.load_resolved(),
               "add quarantine périme le tombstone résolu")

    print("\nTOUS LES TESTS PASSENT ✅")


if __name__ == "__main__":
    try:
        run()
    except AssertionError as e:
        print(f"\n❌ ÉCHEC : {e}")
        sys.exit(1)
