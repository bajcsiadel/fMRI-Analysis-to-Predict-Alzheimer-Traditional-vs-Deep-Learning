"""
Convert ROI signal files
========================

Convert the ROI signal files (.txt) to numpy files (.npz)

Arguments
=========

-h, --help            show this help message and exit
-d DIR, --dir DIR     Directory containing the ROI signals
-o OUT_DIR, --out-dir OUT_DIR
                    Output directory
-e REGEX, --regex REGEX
                    Regular expression to match ROI signal files
"""
import argparse
import warnings

from pathlib import Path

import numpy as np

from tqdm import tqdm

from utils.environment import get_env

parser = argparse.ArgumentParser(__file__, "Convert ROI signals stored in .txt to .npz")
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
    default=r"ROISignals_*.txt",
    help="Regular expression to match ROI signal files",
)


args = parser.parse_args()

args.dir = args.dir or Path(get_env("DATA_ROOT"))

if args.out_dir is None:
    args.out_dir = args.dir

rows, columns = None, None
signal_files = [_ for _ in args.dir.rglob(args.regex)]
for signal_file in tqdm(signal_files, desc="Converting ROI signals"):
    array = np.loadtxt(signal_file)
    if rows is None:
        rows, columns = array.shape
    elif (rows, columns) != array.shape:
        warnings.warn(
            f"Shape of signals do not match. {rows}x{columns} expected "
            f"{array.shape[0]}x{array.shape[1]} got."
        )
    new_file = signal_file.relative_to(args.dir).with_suffix(".npz")
    np.savez(args.out_dir / new_file, signal=array)
