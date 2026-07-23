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
summary_path = root / "csv" / "our_model_fourfold_overall_summary.csv"

with summary_path.open("r", encoding="utf-8", newline="") as handle:
    summary = next(csv.DictReader(handle))
ours = float(summary["mean_held_out_ll_per_trial"])
ours_total_ll = float(summary["total_log_likelihood"])
ours_fold_se = float(summary["fold_standard_error"])

source = html_path.read_text(encoding="utf-8")
marker = '<div class="portable-block portable-layout-full" data-artifact-block-id="cv_block"'
block_start = source.find(marker)
block_end = source.find("</figure></div>", block_start)
if block_start < 0 or block_end < 0:
    raise RuntimeError("Could not locate the primary held-out model comparison block.")
block_end += len("</figure></div>")

replacement = f'''<div class="portable-block portable-layout-full" data-artifact-block-id="cv_block" data-artifact-block-type="chart" data-layout="full"><figure class="portable-content-card portable-chart-summary" data-artifact-id="cv_chart" data-artifact-kind="chart" data-chart-id="cv_chart" data-portable-visual-title="Held out model performance" data-portable-source-host="true" tabindex="0" aria-label="Held out model performance" aria-describedby="portable-source-tooltip-12"><figcaption class="portable-visual-header"><strong>Held out model performance</strong></figcaption><div class="portable-inline-source" data-source-id="cv"><div class="portable-inline-source-content portable-source-tooltip-content" id="portable-source-tooltip-12" role="tooltip"><span class="portable-source-tooltip-heading" aria-hidden="true">Source for Held out model performance</span><strong>Source: Original cross-validation results plus our new four-fold EM-HMM test</strong><span class="portable-source-meta">Original HTML + HMM_fourfold_comparison/csv/our_model_fourfold_overall_summary.csv</span><p class="portable-source-description-data">The original six benchmark values are preserved. Our EM-HMM row is injected from the completed training-only four-fold held-out evaluation.</p></div></div><div class="portable-table-scroll"><table><caption>Held out model performance data</caption><thead><tr><th scope="col">Model</th><th scope="col" class="portable-table-number">Mean held out LL/trial</th></tr></thead><tbody><tr><td>Our EM-HMM (3-state soft-EM)</td><td class="portable-table-number">{ours:.4f}</td></tr><tr><td>Covariate HMM</td><td class="portable-table-number">-0.8133</td></tr><tr><td>Static HMM</td><td class="portable-table-number">-0.8431</td></tr><tr><td>Independent Switching</td><td class="portable-table-number">-0.9052</td></tr><tr><td>Serial stimulus + response</td><td class="portable-table-number">-0.9100</td></tr><tr><td>Serial response</td><td class="portable-table-number">-0.9100</td></tr><tr><td>Serial stimulus</td><td class="portable-table-number">-0.9100</td></tr></tbody></table></div><p class="portable-table-note"><strong>Formal comparison update:</strong> our EM-HMM point estimate is {ours:.4f} LL/trial, 0.0027 above the source Covariate HMM benchmark. Original fold IDs are unavailable, so this point difference is not a paired same-fold significance test.</p></figure></div>'''
source = source[:block_start] + replacement + source[block_end:]

source = source.replace(
    "The best held-out predictor was the Covariate HMM at -0.8133 LL/trial.",
    f"After formally injecting our new four-fold result into the primary comparison, our EM-HMM has the highest point estimate at {ours:.4f} LL/trial, narrowly above the Covariate HMM at -0.8133.",
)
source = source.replace(
    "<p><strong>Prediction.</strong> Covariate HMM had the highest mean held out log likelihood per trial (-0.8133).</p>",
    f"<p><strong>Prediction.</strong> Our EM-HMM has the highest held-out point estimate ({ours:.4f} LL/trial), followed closely by the source Covariate HMM (-0.8133). Because the source fold IDs are unavailable, this is a benchmark comparison rather than a paired same-fold significance test.</p>",
)
result_start = source.find("<p><strong>Result.</strong> Covariate HMM is best at")
if result_start >= 0:
    result_end = source.find("</p>", result_start) + len("</p>")
    source = source[:result_start] + (
        f"<p><strong>Result.</strong> Our EM-HMM has the highest point estimate at {ours:.4f} LL/trial; "
        "the source Covariate HMM is next at -0.8133.</p>"
    ) + source[result_end:]
source = source.replace(
    "<p><strong>Caution.</strong> Absolute LL values depend on response-density units; compare models on the same trials</p>",
    "<p><strong>Caution.</strong> All values use the same response-density units and dataset, but our deterministic folds are newly generated because the source fold IDs were not available. Treat the 0.0027 difference from Covariate HMM as a point comparison, not a paired significance result.</p>",
)
source = source.replace(
    "<td>Covariate HMM is best at -0.8133 LL/trial</td><td>Absolute LL values depend on response-density units; compare models on the same trials</td>",
    f"<td>Our EM-HMM has the highest point estimate at {ours:.4f}; Covariate HMM is -0.8133</td><td>Our folds were newly generated because the original fold IDs were unavailable; the comparison is not paired</td>",
)

# The interactive reader ignores the fallback table above and renders from this
# gzip/base64 JSON payload. Update that authoritative runtime data as well.
payload_pattern = re.compile(
    r'(<template id="data-analytics-portable-artifact-payload-source"[^>]*>)(.*?)(</template>)',
    flags=re.DOTALL,
)
payload_match = payload_pattern.search(source)
if not payload_match:
    raise RuntimeError("Could not locate the embedded artifact payload.")
payload = json.loads(
    gzip.decompress(base64.b64decode(re.sub(r"\s+", "", payload_match.group(2)))).decode("utf-8")
)

cv_summary = payload["snapshot"]["datasets"]["cv_summary"]
cv_summary[:] = [row for row in cv_summary if row.get("model") != "Our_EM_HMM_3state_soft_EM"]
for row in cv_summary:
    row["delta_from_best_per_trial"] = float(row["mean_test_ll_per_trial"]) - ours
cv_summary.insert(
    0,
    {
        "delta_from_best_per_trial": 0.0,
        "folds": 4,
        "mean_test_ll": ours_total_ll / 4.0,
        "mean_test_ll_per_trial": ours,
        "model": "Our_EM_HMM_3state_soft_EM",
        "model_label": "Our EM-HMM",
        "se_test_ll_per_trial": ours_fold_se,
    },
)

for block in payload["manifest"].get("blocks", []):
    if block.get("id") == "one_minute":
        block["body"] = block["body"].replace(
            "The best held-out predictor was the Covariate HMM at -0.8133 LL/trial.",
            f"Our newly injected four-fold EM-HMM result has the highest point estimate at {ours:.4f} LL/trial, narrowly above the Covariate HMM at -0.8133.",
        )
    elif block.get("id") == "technical_summary":
        block["body"] = block["body"].replace(
            "**Prediction.** Covariate HMM had the highest mean held out log likelihood per trial (-0.8133).",
            f"**Prediction.** Our EM-HMM has the highest held-out point estimate ({ours:.4f} LL/trial), followed closely by the source Covariate HMM (-0.8133). This is not a paired same-fold significance test because the source fold IDs are unavailable.",
        )
    elif block.get("id") == "guide_cv_chart":
        block["body"] = (
            "### How to read: Absolute held-out performance\n\n"
            "**Question.** Which model best predicts unseen run sequences?\n\n"
            "**Axes and layout.** Y lists models; x is mean held-out LL/trial\n\n"
            "**Marks and colors.** Bars closer to zero indicate higher predictive density\n\n"
            f"**Result.** Our EM-HMM has the highest point estimate at {ours:.4f} LL/trial; Covariate HMM is next at -0.8133.\n\n"
            "**Caution.** Our folds were newly generated because the original fold IDs were unavailable, so the 0.0027 difference is not a paired significance result."
        )

for chart in payload["manifest"].get("charts", []):
    if chart.get("id") == "cv_chart":
        chart["subtitle"] = "Original source benchmarks plus our training-only four-fold EM-HMM result; higher is better."

for row in payload["snapshot"]["datasets"].get("figure_guide", []):
    if any(str(value) == "Absolute held-out performance" for value in row.values()):
        for key in list(row):
            normalized = key.lower()
            if normalized == "takeaway":
                row[key] = f"Our EM-HMM has the highest point estimate at {ours:.4f}; Covariate HMM is -0.8133"
            elif normalized == "caveat":
                row[key] = "Original fold IDs were unavailable, so the EM-HMM comparison is not paired"

encoded_payload = base64.b64encode(
    gzip.compress(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), mtime=0)
).decode("ascii")
source = source[:payload_match.start(2)] + encoded_payload + source[payload_match.end(2):]

html_path.write_text(source, encoding="utf-8")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

manifest_rows = []
for path in sorted(p for p in root.rglob("*") if p.is_file()):
    if path.name == "output_manifest.csv" or "__pycache__" in path.parts:
        continue
    manifest_rows.append(
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    )
with (root / "logs" / "output_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256"])
    writer.writeheader()
    writer.writerows(manifest_rows)

print(f"Injected Our EM-HMM ({ours:.4f}) into the primary held-out chart/table: {html_path}")
