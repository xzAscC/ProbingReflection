"""Merge judge sidecar verdicts into each trajectory JSON.

After 3 independent judge runs write _judge_run_{1,2,3}.json (each a list of
{problem_id, label, confidence, reflection_instances, rationale}), this script
appends judge_run_1/2/3 keys to every data/hard_negatives/traj_XX.json.

A hard negative is CORRECT for a given run iff label != GENUINE_SELF_REFLECTION
(i.e. SURFACE_REFLECTION or NO_REFLECTION).

Usage:
    uv run python scripts/merge_judge_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "hard_negatives"
RUN_FILES = [
    DATA_DIR / "_judge_run_1.json",
    DATA_DIR / "_judge_run_2.json",
    DATA_DIR / "_judge_run_3.json",
]
TRAJ_GLOB = "traj_*.json"


def load_run(path: Path) -> dict[int, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing judge run file: {path}")
    data = json.loads(path.read_text())
    indexed: dict[int, dict] = {}
    for entry in data:
        pid = entry.get("problem_id")
        if pid is None:
            raise ValueError(f"Entry in {path} missing problem_id: {entry}")
        indexed[int(pid)] = entry
    return indexed


def main() -> None:
    runs = [load_run(p) for p in RUN_FILES]
    traj_files = sorted(DATA_DIR.glob(TRAJ_GLOB))
    if not traj_files:
        raise SystemExit(f"No traj_*.json files found in {DATA_DIR}")

    n_correct_per_run = [0, 0, 0]
    n_all_correct = 0
    n_majority_correct = 0
    label_counts: dict[str, int] = {}

    for tf in traj_files:
        traj = json.loads(tf.read_text())
        pid = int(traj["problem_id"])
        verdicts = []
        for i, run in enumerate(runs):
            if pid not in run:
                raise KeyError(f"problem_id={pid} not found in {RUN_FILES[i].name}")
            v = run[pid]
            # keep only judge-relevant fields, strip problem_id duplication
            verdict = {
                "label": v.get("label", "UNKNOWN"),
                "confidence": v.get("confidence"),
                "reflection_instances": v.get("reflection_instances", []),
                "rationale": v.get("rationale", v.get("reasoning", "")),
            }
            verdicts.append(verdict)
            label = verdict["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
            if label != "GENUINE_SELF_REFLECTION":
                n_correct_per_run[i] += 1
        traj["judge_run_1"] = verdicts[0]
        traj["judge_run_2"] = verdicts[1]
        traj["judge_run_3"] = verdicts[2]

        all_non_genuine = all(v["label"] != "GENUINE_SELF_REFLECTION" for v in verdicts)
        genuine_count = sum(1 for v in verdicts if v["label"] == "GENUINE_SELF_REFLECTION")
        if all_non_genuine:
            n_all_correct += 1
        if genuine_count <= 1:  # majority (>=2 of 3) non-genuine
            n_majority_correct += 1

        tf.write_text(json.dumps(traj, indent=2, ensure_ascii=False))

    n_traj = len(traj_files)
    print(f"Merged {n_traj} trajectories with 3 judge runs each.")
    print(
        f"Per-run correct (non-GENUINE): run1={n_correct_per_run[0]}/{n_traj}, "
        f"run2={n_correct_per_run[1]}/{n_traj}, run3={n_correct_per_run[2]}/{n_traj}"
    )
    print(f"All-3-runs correct: {n_all_correct}/{n_traj}")
    print(f"Majority (>=2 of 3) correct: {n_majority_correct}/{n_traj}")
    print("Label distribution across all 3x25=75 judgments:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
