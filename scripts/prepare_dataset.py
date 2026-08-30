"""
Dataset preparation, validation, and split utility for SYNAPS (SIH26147).
"""

import argparse
from pathlib import Path
import json
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import CLASS_NAMES, IQ_ROOT, METADATA_ROOT, WAV_ROOT, PROCESSED_ROOT
from signal_processing.input.validation import validate_dataset_sample
from ai.training.train import find_dataset, build_dataset, stratified_split


def audit_and_prepare_dataset(output_split: bool = False):
    print("=" * 60)
    print("SYNAPS DATASET AUDIT & PREPARATION")
    print("=" * 60)

    dataset_items = find_dataset()
    valid_count = 0
    invalid_count = 0

    print(f"\nValidating {len(dataset_items)} dataset samples...")
    for idx, item in enumerate(dataset_items):
        res = validate_dataset_sample(item["metadata_path"])
        if res["status"] == "VALID":
            valid_count += 1
        else:
            invalid_count += 1
            print(f"[WARN] Invalid sample {item['sample_id']}: {res['issues']}")

    print(f"\nAudit complete: {valid_count} Valid, {invalid_count} Invalid out of {len(dataset_items)} total.")

    if output_split:
        print("\nBuilding train/val/test splits...")
        X, y = build_dataset(dataset_items, sequence_length=1000)
        X_train, y_train, X_val, y_val, X_test, y_test = stratified_split(X, y)
        PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

        split_summary = {
            "total_samples": len(dataset_items),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "feature_shape": list(X_train.shape[1:]),
            "classes": CLASS_NAMES,
        }
        with open(PROCESSED_ROOT / "split_summary.json", "w", encoding="utf-8") as f:
            json.dump(split_summary, f, indent=2)

        print(f"Split summary saved to: {PROCESSED_ROOT / 'split_summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit and prepare dataset.")
    parser.add_argument("--save-splits", action="store_true", help="Save split summary")
    args = parser.parse_args()

    audit_and_prepare_dataset(output_split=args.save_splits)