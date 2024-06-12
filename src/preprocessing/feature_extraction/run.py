"""
Extract features
================

Extracting features from the ROI signals.
"""
import argparse
import warnings

from pathlib import Path

import numpy as np
from icecream import ic

from tqdm import tqdm

from utils.environment import get_env

parser = argparse.ArgumentParser(__file__, "Extract features from ROI signals")
parser.add_argument(
    "-d", "--dir",
    type=Path,
    help="Directory containing the ROI signals",
)
parser.add_argument(
    "-o", "--out-dir",
    type=Path,
    help="Output directory",
)
parser.add_argument(
    "-e", "--regex",
    type=str,
    default=r"ROISignals_*.npz",
    help="Regular expression to match ROI signal files",
)

args = parser.parse_args()

args.dir = args.dir or Path(get_env("DATA_ROOT"))

if args.out_dir is None:
    args.out_dir = args.dir

signal_files = [_ for _ in args.dir.rglob(args.regex)]
for signal_file in tqdm(signal_files, desc="Extracting featured from ROI signals"):
    signal = np.load(signal_file)["signal"]

