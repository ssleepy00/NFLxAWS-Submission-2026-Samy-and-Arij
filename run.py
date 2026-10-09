"""End-to-end: build frames -> fit PCA+Ridge and PCA+Lasso -> evaluate -> figures.

Usage:  .venv/bin/python run.py [--rebuild]
"""
import json
import sys

from pocket_safety import data, model, plots

if __name__ == "__main__":
    data.build_all_frames(force="--rebuild" in sys.argv)
    out = model.train_and_evaluate()
    plots.make_all(out)
    print(json.dumps(out["results"], indent=2))
