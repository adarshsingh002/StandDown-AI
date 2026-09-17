#!/usr/bin/env python3
"""
Packaging script for HackerRank Orchestrate "Buy or Wait?" submission.
Creates code.zip containing:
- code/ (with main.py, evaluation/ folder, cache)
- evaluation/ (with usage_report.md)
- README.md
- requirements.txt
"""

import zipfile
import os
from pathlib import Path

def create_code_zip():
    repo_root = Path(__file__).resolve().parent
    zip_path = repo_root / "code.zip"

    items_to_include = [
        "code",
        "evaluation",
        "README.md",
        "README_WORKFLOW.md",
        "requirements.txt",
    ]

    print(f"Packaging {zip_path.name} from {repo_root}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item_name in items_to_include:
            item_path = repo_root / item_name
            if not item_path.exists():
                print(f"Warning: {item_name} does not exist, skipping.")
                continue

            if item_path.is_file():
                zf.write(item_path, arcname=item_name)
                print(f"  Added file: {item_name}")
            elif item_path.is_dir():
                for root, dirs, files in os.walk(item_path):
                    # Skip __pycache__ and .venv
                    dirs[:] = [d for d in dirs if d not in ("__pycache__", ".venv", ".git")]
                    for file in files:
                        if file.endswith(".pyc"):
                            continue
                        full_path = Path(root) / file
                        arcname = full_path.relative_to(repo_root)
                        zf.write(full_path, arcname=str(arcname))
                        print(f"  Added: {arcname}")

    print(f"Successfully generated {zip_path} ({zip_path.stat().st_size:,} bytes)")

if __name__ == "__main__":
    create_code_zip()
