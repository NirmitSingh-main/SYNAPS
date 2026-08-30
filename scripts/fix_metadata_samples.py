from pathlib import Path
import json

from ai.preprocessing.iq_loader import load_iq_file

IQ_ROOT = Path("data/iq")
METADATA_ROOT = Path("data/metadata")


def find_metadata_path(iq_path: Path):
    candidates = []

    base = iq_path.stem
    candidates.append(base + ".json")

    # Handle naming differences between old and new dataset conventions.
    if "_qam16" in base:
        candidates.append(base.replace("_qam16", "_16qam") + ".json")
    if "_16qam" in base:
        candidates.append(base.replace("_16qam", "_qam16") + ".json")

    if "_2fsk" in base:
        candidates.append(base.replace("_2fsk", "_fsk") + ".json")
    if "_fsk" in base:
        candidates.append(base.replace("_fsk", "_2fsk") + ".json")

    seen = set()
    ordered = []
    for name in candidates:
        if name not in seen:
            seen.add(name)
            ordered.append(name)

    for name in ordered:
        matches = list(METADATA_ROOT.rglob(name))
        if matches:
            return matches[0]

    return None


def main():
    updated = 0
    missing = 0

    for iq_path in sorted(IQ_ROOT.rglob("*.iq")):
        metadata_path = find_metadata_path(iq_path)

        if metadata_path is None:
            print(f"WARNING: metadata not found for: {iq_path}")
            missing += 1
            continue

        iq = load_iq_file(iq_path)
        actual_samples = len(iq)

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        old_samples = metadata.get("samples")

        if old_samples != actual_samples:
            metadata["samples"] = actual_samples

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
                f.write("\n")

            updated += 1

            print(
                f"UPDATED: {metadata_path} "
                f"{old_samples} -> {actual_samples}"
            )

    print()
    print("==============================")
    print("METADATA SAMPLE FIX")
    print("==============================")
    print("Updated:", updated)
    print("Missing:", missing)


if __name__ == "__main__":
    main()
