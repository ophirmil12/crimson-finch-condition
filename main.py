"""
Main runner — executes all analysis steps in order.
Run from the project root:  python main.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

STEPS = [
    ROOT / "01_data_exploration" / "01_explore.py",
    ROOT / "02_pca"              / "02_pca.py",
    ROOT / "03_glm"              / "03_glm.py",
    ROOT / "04_visualization"    / "04_visualization.py",
    ROOT / "05_year_analysis"    / "05_year_analysis.py",
    ROOT / "06_ml"               / "06_ml.py",
    ROOT / "07_sex_interaction"  / "07_sex_interaction.py",
    ROOT / "08_dimred_clustering" / "08_dimred_clustering.py",
]

def run_step(script: Path):
    print(f"\n{'='*60}")
    print(f"  Running: {script.relative_to(ROOT)}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed: {script.name}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    for step in STEPS:
        run_step(step)
    print("\n" + "=" * 60)
    print("  All steps complete.")
    print(f"  Figures → {ROOT / 'figures'}")
    print(f"  Results → {ROOT / 'results'}")
    print("=" * 60)
