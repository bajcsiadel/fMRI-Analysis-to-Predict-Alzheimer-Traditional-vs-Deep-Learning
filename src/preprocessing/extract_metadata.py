"""
Extract metadata
================

Customize metadata.

Arguments
=========

-h, --help            show this help message and exit
-f FILE, --file FILE  Name of the original metadata file
-d DIR, --dir DIR     Directory of the dataset
-o OUT_DIR, --out-dir OUT_DIR
                    Output directory
--out-file OUT_FILE   Name of the output file
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from utils.environment import get_env


parser = argparse.ArgumentParser(__file__, "Customize metadata")
parser.add_argument(
    "-f", "--file",
    type=str,
    default="original-metadata.csv",
    help="Name of the original metadata file",
)
parser.add_argument(
    "-d", "--dir",
    type=Path,
    help="Directory of the dataset",
)
parser.add_argument(
    "-o", "--out-dir",
    type=Path,
    help="Output directory",
)
parser.add_argument(
    "--out-file",
    type=str,
    default="metadata.csv",
    help="Name of the output file"
)


args = parser.parse_args()

DATA_ROOT = args.dir or Path(get_env("DATA_ROOT"))
if DATA_ROOT.name != "ADNI":
    DATA_ROOT = DATA_ROOT / "ADNI"
OUTPUT_DIR = args.out_dir or DATA_ROOT
OUTPUT_DIR.mkdir(exist_ok=True)

logger = logging.Logger(__name__)
logger.setLevel(logging.NOTSET)

formatter = logging.Formatter(
    "[%(asctime)s][%(levelname)-7s] %(message)s",
    "%Y-%m-%d %H:%M:%S"
)

file_handler = logging.FileHandler(f"log_{Path(__file__).stem}.txt")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

metadata = pd.read_csv(DATA_ROOT / args.file, parse_dates=["Acq Date"])
metadata = metadata[metadata["Modality"] == "fMRI"]

restructured_metadata = pd.DataFrame(
    columns=["filename", "patient_id", "session_time", "label", "sex", "age"]
)
restructured_metadata["patient_id"] = metadata["Subject"]
restructured_metadata["session_time"] = metadata["Acq Date"].apply(
    lambda date: date.strftime("%Y_%m_%d")
)
restructured_metadata["filename"] = list(
    map(lambda row: f"{row[1]}-{row[2]}", restructured_metadata.values)
)
restructured_metadata["label"] = metadata["Group"]
restructured_metadata["sex"] = metadata["Sex"]
restructured_metadata["age"] = metadata["Age"]

restructured_metadata = restructured_metadata.sort_values("filename")

restructured_metadata.to_csv(OUTPUT_DIR / args.out_file, index=False)

logger.info(f"Metadata saved to {OUTPUT_DIR / args.out_file}")
