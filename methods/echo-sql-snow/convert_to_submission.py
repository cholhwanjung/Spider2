"""Copy per-instance result.csv files into the spider2-snow evaluator layout.

Usage:
  uv run python convert_to_submission.py \
      --input results/{model}_{suffix} \
      --output ../../spider2-snow/evaluation_suite/{suffix}
"""
import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="results/{model}_{suffix} folder")
    parser.add_argument("--output", required=True, help="evaluation_suite/{suffix} folder")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for inst_dir in sorted(src.iterdir()):
        if not inst_dir.is_dir():
            continue
        result_csv = inst_dir / "result.csv"
        if not result_csv.exists() or result_csv.stat().st_size == 0:
            skipped += 1
            continue
        target = dst / f"{inst_dir.name}.csv"
        shutil.copy(result_csv, target)
        written += 1

    print(f"Wrote {written} CSVs to {dst}, skipped {skipped} empty/missing.")


if __name__ == "__main__":
    main()
