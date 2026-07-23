from __future__ import annotations

import json
import os
import sys
from pathlib import Path


workspace = Path(__file__).resolve().parents[2]
runtime = workspace / "tmp" / "jupyter-notebook" / "runtime"
sys.path.insert(0, str(runtime))

import nbformat
from nbclient import NotebookClient


python_executable = Path(
    r"C:\Users\baoba\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
)
kernel_root = workspace / "tmp" / "jupyter-notebook" / "kernels"
kernel_dir = kernel_root / "hmm-fourfold"
kernel_dir.mkdir(parents=True, exist_ok=True)
kernel_spec = {
    "argv": [str(python_executable), "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "HMM four-fold temporary kernel",
    "language": "python",
    "env": {
        "PYTHONPATH": str(runtime),
        "MPLBACKEND": "Agg",
        "IPYTHONDIR": str(workspace / "tmp" / "jupyter-notebook" / "ipython"),
    },
}
(kernel_dir / "kernel.json").write_text(json.dumps(kernel_spec, indent=2), encoding="utf-8")
existing_jupyter_path = os.environ.get("JUPYTER_PATH", "")
os.environ["JUPYTER_PATH"] = str(kernel_root.parent) + (os.pathsep + existing_jupyter_path if existing_jupyter_path else "")
os.environ["PYTHONPATH"] = str(runtime)
os.environ["MPLBACKEND"] = "Agg"
os.environ["IPYTHONDIR"] = str(workspace / "tmp" / "jupyter-notebook" / "ipython")

notebook_path = workspace / "HMM_fourfold_comparison" / "notebooks" / "HMM_fourfold_heldout_comparison.ipynb"
notebook = nbformat.read(notebook_path, as_version=4)
client = NotebookClient(
    notebook,
    timeout=None,
    kernel_name="hmm-fourfold",
    resources={"metadata": {"path": str(workspace)}},
    allow_errors=False,
)
client.execute(cwd=str(workspace))
nbformat.write(notebook, notebook_path)

code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
error_outputs = [
    output
    for cell in code_cells
    for output in cell.get("outputs", [])
    if output.get("output_type") == "error"
]
print(
    json.dumps(
        {
            "notebook": str(notebook_path),
            "code_cells": len(code_cells),
            "execution_counts": [cell.get("execution_count") for cell in code_cells],
            "error_outputs": len(error_outputs),
        },
        indent=2,
    )
)
