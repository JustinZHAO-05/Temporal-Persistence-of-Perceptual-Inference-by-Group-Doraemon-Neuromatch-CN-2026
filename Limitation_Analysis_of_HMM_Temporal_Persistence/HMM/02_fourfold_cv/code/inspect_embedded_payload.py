from __future__ import annotations

import base64
import gzip
import json
import re
from pathlib import Path


html_path = Path(__file__).resolve().parents[1] / "html" / "HMM_fourfold_comparison_with_original_report.html"
text = html_path.read_text(encoding="utf-8")
match = re.search(
    r'<template id="data-analytics-portable-artifact-payload-source"[^>]*>(.*?)</template>',
    text,
    flags=re.DOTALL,
)
if not match:
    raise RuntimeError("Payload template not found")
payload_text = gzip.decompress(base64.b64decode(re.sub(r"\s+", "", match.group(1)))).decode("utf-8")
print("payload_chars", len(payload_text))
print("our_emhmm_occurrences", payload_text.count("Our EM-HMM"))
print("heldout_title_occurrences", payload_text.count("Held out model performance"))
print("covariate_occurrences", payload_text.count("Covariate HMM"))

try:
    payload = json.loads(payload_text)
except json.JSONDecodeError:
    payload = None
print("json_type", type(payload).__name__)
if isinstance(payload, dict):
    print("top_keys", sorted(payload.keys()))
    print("top_types", {key: type(value).__name__ for key, value in payload.items()})
    print("package_info", payload.get("package_info"))
    print("packageInfo", payload.get("packageInfo"))
    print("ok", payload.get("ok"), "surface", payload.get("surface"), "widget_type", payload.get("widget_type"))
    print("manifest_keys", sorted(payload.get("manifest", {}).keys()))
    snapshot = payload.get("snapshot", {})
    print("snapshot_keys", sorted(snapshot.keys()) if isinstance(snapshot, dict) else type(snapshot).__name__)
    if isinstance(snapshot, dict):
        datasets = snapshot.get("datasets", {})
        print("datasets_type", type(datasets).__name__)
        if isinstance(datasets, dict):
            print("dataset_keys", sorted(datasets.keys()))
            print("cv_summary", json.dumps(datasets.get("cv_summary"), ensure_ascii=False, indent=2)[:12000])

    def walk(value, path="root"):
        if isinstance(value, dict):
            if value.get("id") in {"one_minute", "technical_summary", "guide_cv_chart", "cv_chart", "cv_block"}:
                print("OBJECT_PATH", path, "ID", value.get("id"), "KEYS", sorted(value.keys()))
            for key, child in value.items():
                if key == "cv_summary":
                    print("CV_SUMMARY_PATH", f"{path}.{key}", "TYPE", type(child).__name__)
                walk(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
    walk(payload)

for term in ("Held out model performance", "cv_chart", "Covariate HMM"):
    start = 0
    positions = []
    while True:
        position = payload_text.find(term, start)
        if position < 0:
            break
        positions.append(position)
        start = position + 1
    print("TERM", term, "POSITIONS", positions)
    for position in positions[:3]:
        print(payload_text[max(0, position - 500): position + 1200])
