"""End-to-end: build frames -> fit PCA+Ridge and PCA+Lasso -> evaluate -> figures.

Usage:
    python run.py [--rebuild] [--data-dir PATH]

PATH can be the dataset root, its data/ folder, or set via $POCKET_DATA_DIR.
"""
import argparse
import json
import sys

from pocket_safety import config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true", help="recompute judge/outputs/frames.parquet")
    ap.add_argument("--data-dir", help="folder containing plays.csv, pffScoutingData.csv and tracking/")
    args = ap.parse_args()

    data_dir = config.set_data_dir(args.data_dir)

    from pocket_safety import data, model, plots  # import after the data dir is set

    try:
        game_ids = data.check_data_dir()
    except data.DataNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"Using data in {data_dir} ({len(game_ids)} games)")

    data.build_all_frames(force=args.rebuild)
    out = model.train_and_evaluate()
    plots.make_all(out)
    print(json.dumps(out["results"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
