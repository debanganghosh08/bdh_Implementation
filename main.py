# main.py
# ============================
# End-to-End BDH Pipeline
# ============================

import subprocess
import sys
from pathlib import Path


def run_step(step_name, command):
    print(f"\n🚀 Running: {step_name}")
    print("-" * 60)

    result = subprocess.run(
        [sys.executable] + command,
        capture_output=False
    )

    if result.returncode != 0:
        print(f"\n❌ Failed at step: {step_name}")
        sys.exit(1)

    print(f"✅ Completed: {step_name}")


def main():
    root = Path(__file__).parent

    # Safety checks
    required_files = [
        "extract_tension_train.py",
        "train_head.py",
        "run_bdh.py"
    ]

    for f in required_files:
        if not (root / f).exists():
            print(f"❌ Missing required file: {f}")
            sys.exit(1)

    print("\n==============================")
    print("🧠 BDH Narrative Reasoning Pipeline")
    print("==============================")

    # Step 1: Extract tension features from train.csv
    run_step(
        "Extract BDH tension features (train)",
        ["extract_tension_train.py"]
    )

    # Step 2: Train tension classifier head
    run_step(
        "Train tension classifier head",
        ["train_head.py"]
    )

    # Step 3: Run inference on test.csv
    run_step(
        "Run BDH inference on test set",
        ["run_bdh.py"]
    )

    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY")
    print("📄 Output file: results.csv")


if __name__ == "__main__":
    main()
