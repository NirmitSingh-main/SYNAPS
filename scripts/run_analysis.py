"""
End-to-end signal intelligence analysis CLI script for SYNAPS (SIH26147).
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.pipeline import analyze_signal
from intelligence.report.report import format_text_report, save_report
from project_paths import RESULT_ROOT


def main():
    parser = argparse.ArgumentParser(description="SYNAPS End-to-End Signal Intelligence Analysis")
    parser.add_argument("file_path", type=str, help="Path to input .iq or .wav signal")
    parser.add_argument("--sample-rate", type=float, default=None, help="Sampling frequency in Hz")
    parser.add_argument("--sps", type=int, default=10, help="Samples per symbol")
    parser.add_argument("--save", action="store_true", help="Save analysis report to disk")
    parser.add_argument("--output-dir", type=str, default=str(RESULT_ROOT / "reports"), help="Output directory")
    args = parser.parse_args()

    print(f"Running full intelligence pipeline on: {args.file_path} ...\n")
    results = analyze_signal(
        file_path_or_samples=args.file_path,
        sample_rate=args.sample_rate,
        samples_per_symbol=args.sps,
    )

    report_dict = results["report"]
    formatted_text = format_text_report(report_dict)
    print(formatted_text)

    if args.save:
        out_dir = Path(args.output_dir)
        stem = Path(args.file_path).stem
        json_p, txt_p = save_report(report_dict, output_dir=out_dir, base_name=f"report_{stem}")
        print(f"\nSaved reports:\n  JSON: {json_p}\n  Text: {txt_p}")


if __name__ == "__main__":
    main()