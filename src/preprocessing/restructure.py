import argparse
import copy
import json
import logging
import os
import shutil
import sys
import time
import warnings
from collections import OrderedDict
from enum import Enum
from pathlib import Path

import datetime

import numpy as np
import pandas as pd
import pydicom
from dicom2nifti.convert_dicom import dicom_array_to_nifti
from icecream import ic

from utils.environment import get_env
from utils import pipe


parser = argparse.ArgumentParser(__file__, "Restructure NIFTI data")
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


args = parser.parse_args()

DATA_ROOT = args.dir or Path(get_env("DATA_ROOT"))
OUTPUT_DIR = args.out_dir or (DATA_ROOT / ".." / "structured")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")

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

DATA_FILE = DATA_ROOT / "data.csv"

if DATA_FILE.exists():
    data = pd.read_csv(DATA_FILE, index_col=0, header=0)
else:
    raise FileNotFoundError(f"data.csv not found in {DATA_ROOT} directory!")

rs_fmri_data = data[data["image_type"] == "Resting State fMRI"]

logger.info(f"Number of image fMRIs in ADNI dataset: {len(rs_fmri_data)}")
for index, (_, details) in enumerate(rs_fmri_data.iterrows(), start=1):
    iteration = f"{index} / {len(rs_fmri_data)}"
    session_time = details["session_time"].split("_")[0].replace("-", "_")
    patient_id = details["patient_id"]

    dir_name = f"{patient_id}-{session_time}"

    logger.info(f"{iteration} {dir_name = }")

    sub_dir_name = f"{details['direction_z']}_{details['TR']}"

    result_sub_dir_path = OUTPUT_DIR / sub_dir_name
    if not result_sub_dir_path.exists():
        result_sub_dir_path.mkdir(parents=True)
        (result_sub_dir_path / "FunRaw").mkdir()
        (result_sub_dir_path / "T1Raw").mkdir()

    shutil.copytree(DATA_ROOT / dir_name / "functional", OUTPUT_DIR / sub_dir_name / "FunRaw" / dir_name)
    shutil.copytree(DATA_ROOT / dir_name / "structural", OUTPUT_DIR / sub_dir_name / "T1Raw" / dir_name)
