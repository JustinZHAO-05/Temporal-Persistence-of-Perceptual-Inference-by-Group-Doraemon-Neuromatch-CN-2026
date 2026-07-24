from __future__ import annotations

import argparse
import json
from pathlib import Path

from perceptual_arbitration.data import load_direction_data
from perceptual_arbitration.publication import render_publication_report
from perceptual_arbitration.run_metadata import atomic_write_json, update_stage


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Render publication tables, figures, and HTML from completed analysis outputs.")
    parser.add_argument("--csv", default="data/data01_direction4priors.csv")
    parser.add_argument("--out", default="outputs/full_run")
    parser.add_argument("--package-html", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    csv_path = _resolve(project_root, args.csv)
    out_dir = _resolve(project_root, args.out)
    data = load_direction_data(csv_path)
    result = render_publication_report(data, out_dir, package_html=bool(args.package_html))
    manifest_path = out_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        update_stage(
            manifest,
            "publication_report",
            "complete",
            report_status=result["status"],
            figures=len(result["figures"]),
            html=str(result["html_path"]) if result["html_path"] else None,
        )
        manifest["publication_ready"] = bool(
            manifest.get("publication_ready", False) and result["status"] == "ready"
        )
        manifest.setdefault("validation", {})["report_issues"] = result["issues"]
        atomic_write_json(manifest, manifest_path)
    print(json.dumps({
        "status": result["status"],
        "issues": result["issues"],
        "figures": len(result["figures"]),
        "summary": str(result["summary_path"]),
        "artifact": str(result["artifact_path"]),
        "html": str(result["html_path"]) if result["html_path"] else None,
        "receipt": result["receipt"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
