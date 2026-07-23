from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path


workspace = Path(__file__).resolve().parents[2]
root = workspace / "HMM_fourfold_comparison"
html_path = root / "html" / "HMM_fourfold_comparison_with_original_report.html"
source_html_path = workspace / "0_compare" / "perceptual_arbitration_results.html"
text = html_path.read_text(encoding="utf-8")
source_text = source_html_path.read_text(encoding="utf-8")

marker = 'data-artifact-block-id="cv_block"'
start = text.find(marker)
end = text.find("</figure></div>", start) + len("</figure></div>")
block = text[start:end]

payload_match = re.search(
    r'<template id="data-analytics-portable-artifact-payload-source"[^>]*>(.*?)</template>',
    text,
    flags=re.DOTALL,
)
if not payload_match:
    raise RuntimeError("Embedded runtime payload not found")
payload = json.loads(
    gzip.decompress(base64.b64decode(re.sub(r"\s+", "", payload_match.group(1)))).decode("utf-8")
)
runtime_cv = payload["snapshot"]["datasets"]["cv_summary"]
runtime_our_rows = [row for row in runtime_cv if row.get("model") == "Our_EM_HMM_3state_soft_EM"]
runtime_guide = [
    block for block in payload["manifest"].get("blocks", []) if block.get("id") == "guide_cv_chart"
]

before = json.loads((root / "model_artifacts" / "protected_snapshot_before.json").read_text(encoding="utf-8"))
after = json.loads((root / "model_artifacts" / "protected_snapshot_after.json").read_text(encoding="utf-8"))

checks = {
    "primary_cv_block_found": start >= 0 and end > start,
    "our_emhmm_in_primary_block": "Our EM-HMM (3-state soft-EM)" in block,
    "our_value_in_primary_block": "-0.8106" in block,
    "seven_models_in_primary_table": block.count("<tr><td>") == 7,
    "covariate_value_preserved": "Covariate HMM</td><td class=\"portable-table-number\">-0.8133" in block,
    "static_value_preserved": "Static HMM</td><td class=\"portable-table-number\">-0.8431" in block,
    "main_summary_updated": "our EM-HMM has the highest point estimate at -0.8106" in text,
    "technical_summary_updated": "Our EM-HMM has the highest held-out point estimate (-0.8106 LL/trial)" in text,
    "guide_result_updated": "Our EM-HMM has the highest point estimate at -0.8106 LL/trial" in text,
    "comparison_caveat_present": "not a paired same-fold significance test" in block,
    "runtime_payload_has_one_emhmm_row": len(runtime_our_rows) == 1,
    "runtime_payload_emhmm_is_first": runtime_cv[0].get("model") == "Our_EM_HMM_3state_soft_EM",
    "runtime_payload_emhmm_value": abs(float(runtime_our_rows[0]["mean_test_ll_per_trial"]) + 0.8105676233398648) < 1e-12,
    "runtime_payload_has_seven_models": len(runtime_cv) == 7,
    "runtime_guide_updated": len(runtime_guide) == 1 and "Our EM-HMM has the highest point estimate" in runtime_guide[0].get("body", ""),
    "original_source_html_not_injected": "Our EM-HMM (3-state soft-EM)" not in source_text,
    "protected_result_folders_unchanged": before == after,
}

lines = ["PRIMARY HELD-OUT CHART EM-HMM INJECTION VALIDATION"]
lines += [f"{name}: {'PASS' if passed else 'FAIL'}" for name, passed in checks.items()]
lines.append(f"overall_status: {'PASS' if all(checks.values()) else 'FAIL'}")
report_path = root / "logs" / "PRIMARY_INJECTION_VALIDATION.txt"
report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

rows = []
for path in sorted(p for p in root.rglob("*") if p.is_file()):
    if path.name == "output_manifest.csv" or "__pycache__" in path.parts:
        continue
    rows.append(
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    )
with (root / "logs" / "output_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256"])
    writer.writeheader()
    writer.writerows(rows)

print("\n".join(lines))
if not all(checks.values()):
    raise SystemExit(1)
