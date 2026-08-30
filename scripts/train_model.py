"""
Model training CLI script for SYNAPS (SIH26147).
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.training.train import run_training_pipeline


def main():
    parser = argparse.ArgumentParser(description="SYNAPS AI Model Training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--smoke-test", action="store_true", help="Run 2-epoch smoke test without overwriting best weights")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset size for rapid testing")
    args = parser.parse_args()

    print("========================================")
    print("        SYNAPS MODEL TRAINING")
    print("========================================")
    if args.smoke_test:
        print("[MODE] SMOKE TEST ACTIVATED (2 epochs, test validation)")

    results = run_training_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_samples=args.max_samples,
        save_checkpoint=not args.smoke_test,
        smoke_test=args.smoke_test,
    )

    print("\nTraining summary:")
    print(f"  Best Validation Accuracy : {results['best_val_accuracy'] * 100:.2f}%")
    print(f"  Final Test Accuracy       : {results['test_accuracy'] * 100:.2f}%")
    print(f"  Results saved to          : {results['result_txt']}")


if __name__ == "__main__":
    main()