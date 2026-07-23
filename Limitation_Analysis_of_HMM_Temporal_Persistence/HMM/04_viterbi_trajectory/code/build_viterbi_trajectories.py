"""Create an all-subject Viterbi trajectory overview from the fitted 3-state HMM.

The source CSV files are outputs of the original full-data three-state HMM fit.
They contain a block-wise Viterbi decoding, so no model is refit here.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "HMM" / "01_full_data_fit" / "results"
OUTPUT = Path(__file__).resolve().parents[1] / "results" / "reproduced"

STATE_ORDER = ["sensory", "prior", "lapse"]
STATE_LABEL = {
    "sensory": "Sensory",
    "prior": "Prior",
    "lapse": "Lapse / random",
}
COLORS = ["#2F80ED", "#E67E22", "#8E44AD"]


def read_subject(subject_id: int) -> list[dict[str, str]]:
    path = SOURCE / f"subject_{subject_id:02d}" / f"subject_{subject_id:02d}_trial_posteriors.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"block_id", "trial_index", "viterbi_state"}
    missing = required.difference(rows[0] if rows else [])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if not {row["viterbi_state"] for row in rows}.issubset(STATE_LABEL):
        raise ValueError(f"Unexpected state label in subject {subject_id}")
    previous_block = None
    completed_blocks: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if row["block_id"] != previous_block and row["block_id"] in completed_blocks:
            raise ValueError(f"Subject {subject_id} has a non-contiguous block in the saved trial order")
        if previous_block is not None and row["block_id"] != previous_block:
            completed_blocks.add(previous_block)
        row["subject_id"] = str(subject_id)
        row["global_trial"] = str(index)
        row["is_block_start"] = str(int(row["block_id"] != previous_block))
        previous_block = row["block_id"]
    return rows


def draw_trajectory_overview(subject_frames: list[list[dict[str, str]]]) -> None:
    """Write a dependency-free SVG: contiguous same-state trials are one rectangle."""
    width, left, right = 1600, 165, 55
    top, row_height, gap = 145, 36, 22
    height = top + len(subject_frames) * (row_height + gap) + 92
    plot_width = width - left - right
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text { font-family: Arial, Helvetica, sans-serif; fill:#1f2937; } .small { font-size:14px; } .label { font-size:17px; font-weight:700; } .title { font-size:26px; font-weight:700; }</style>',
        '<text x="800" y="42" text-anchor="middle" class="title">EM HMM: Viterbi trajectories for all subjects</text>',
    ]
    legend_x = 465
    for index, state in enumerate(STATE_ORDER):
        x = legend_x + index * 260
        parts.append(f'<rect x="{x}" y="85" width="20" height="20" rx="3" fill="{COLORS[index]}"/>')
        parts.append(f'<text x="{x + 29}" y="101" class="small">{STATE_LABEL[state]}</text>')

    for subject_number, rows in enumerate(subject_frames, start=1):
        y = top + (subject_number - 1) * (row_height + gap)
        n = len(rows)
        scale = plot_width / n
        parts.append(f'<text x="112" y="{y + 24}" text-anchor="middle" class="label">S{subject_number:02d}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{plot_width}" height="{row_height}" fill="#f3f4f6" stroke="#d1d5db"/>')
        run_start, previous_state = 0, rows[0]["viterbi_state"]
        for trial_index, row in enumerate(rows[1:], start=1):
            current_state = row["viterbi_state"]
            if current_state != previous_state:
                x = left + run_start * scale
                run_width = (trial_index - run_start) * scale
                parts.append(f'<rect x="{x:.3f}" y="{y}" width="{run_width:.3f}" height="{row_height}" fill="{COLORS[STATE_ORDER.index(previous_state)]}"/>')
                run_start, previous_state = trial_index, current_state
        x = left + run_start * scale
        parts.append(f'<rect x="{x:.3f}" y="{y}" width="{(n-run_start)*scale:.3f}" height="{row_height}" fill="{COLORS[STATE_ORDER.index(previous_state)]}"/>')
        for trial_index, row in enumerate(rows):
            if trial_index and row["is_block_start"] == "1":
                x = left + trial_index * scale
                parts.append(f'<line x1="{x:.3f}" y1="{y}" x2="{x:.3f}" y2="{y+row_height}" stroke="white" stroke-width="0.8"/>')
        parts.append(f'<text x="{left + plot_width + 10}" y="{y + 23}" class="small">n = {n:,}</text>')

    parts.append(f'<text x="{left + plot_width/2}" y="{height-25}" text-anchor="middle" class="small">Trial order within subject (blocks concatenated)</text>')
    parts.append('</svg>')
    (OUTPUT / "all_subjects_viterbi_trajectories.svg").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frames = [read_subject(subject_id) for subject_id in range(1, 13)]
    all_trials = [row for frame in frames for row in frame]
    fields = list(all_trials[0])
    with (OUTPUT / "all_subjects_viterbi_states.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_trials)
    draw_trajectory_overview(frames)
    counts = Counter((row["subject_id"], row["viterbi_state"]) for row in all_trials)
    totals = Counter(row["subject_id"] for row in all_trials)
    with (OUTPUT / "viterbi_state_summary_by_subject.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject_id", "viterbi_state", "n_trials", "total_trials", "proportion"])
        writer.writeheader()
        for subject_id in map(str, range(1, 13)):
            for state in STATE_ORDER:
                count = counts[(subject_id, state)]
                writer.writerow({"subject_id": subject_id, "viterbi_state": state, "n_trials": count, "total_trials": totals[subject_id], "proportion": count / totals[subject_id]})
    n_blocks = len({(row["subject_id"], row["block_id"]) for row in all_trials})
    with (OUTPUT / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "n_subjects", "n_trials", "n_blocks", "states"])
        writer.writeheader()
        writer.writerow({"source": "all_subject_results_revised (original full-data 3-state HMM)", "n_subjects": 12, "n_trials": len(all_trials), "n_blocks": n_blocks, "states": "; ".join(STATE_ORDER)})


if __name__ == "__main__":
    main()
