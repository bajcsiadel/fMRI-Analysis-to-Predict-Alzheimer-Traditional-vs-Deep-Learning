"""
Restructure
===========

Restructure images from ADNI dataset to a more convenient format and
creating metadata file.

Arguments
=========

-h, --help            show this help message and exit
-f [FILE ...], --file [FILE ...]
                    Path to dicom file
-d DIR, --dir DIR     Directory of the dataset
-o OUT_DIR, --out-dir OUT_DIR
                    Output directory
--out-file OUT_FILE   Name of the output file
"""
import argparse
import copy
import logging
import shutil
import sys
import typing
import warnings
from enum import Enum
from pathlib import Path

import datetime

import numpy as np
import pandas as pd
import pydicom

from utils.environment import get_env
from utils import pipe


parser = argparse.ArgumentParser(__file__, "Create metadata file")
parser.add_argument(
    "-f", "--file",
    type=Path,
    nargs="*",
    default=[],
    help="Path to dicom file",
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
    default="image-data.csv",
    help="Name of the output file"
)


class MRITags(Enum):
    # same tags for both structural and functional images
    acquisition_time     = (0x00080032, )  # noqa
    patient_id           = (0x00100020, )  # noqa
    patient_sex          = (0x00100040, )  # noqa
    patient_age          = (0x00101010, )  # noqa
    acquisition_type     = (0x00180023, )  # noqa
    slice_thickness      = (0x00180050, 0x00180088)  # noqa
    echo_time            = (0x00180081, )  # noqa
    imaging_frequency    = (0x00180084, )  # noqa
    magnetic_filed_strength = (0x00180087, )  # noqa
    protocol_name        = (0x00181030, )  # noqa
    patient_position     = (0x00185100, )  # noqa
    image_position       = (0x00200032, )  # noqa
    n_frames             = (0x00280008, )  # noqa
    row                  = (0x00280010, )  # noqa
    column               = (0x00280011, )  # noqa
    # different tags for structural and functional images acquired 2D or 3D
    acquisition_contrast = (  # noqa
        0x00089209,
        (0x2005140f, 0, 0x00089209),
        (0x52009230, 0, 0x00189226, 0, 0x00189209),
    )
    repetition_time      = (  # noqa
        0x00180080,
        0x20051030,
        (0x52009229, 0, 0x00189112, 0, 0x00180080)
    )
    number_of_volumes    = (  # noqa
        0x00200105,  # 2D images
        (0x52009230, 0, 0x2005140f, 0, 0x00200105),  # 3D images
    )
    pixel_spacing        = (  # noqa
        0x00280030,  # 2D images
        (0x52009230, 0, 0x00289110, 0, 0x00280030),  # 3D images
    )
    acquisition_duration = (  # noqa
        0x0019105A,  # structural 2D
        0x00189073,  # structural 3D, functional 2D-3D
    )
    number_of_slices_per_volume = (  # noqa
        0x00280008,  # structural 3D
        0x2001102D,  # structural 2D
        0x20011018,  # functional 2D, 3D
    )


def get_tag_from_meta(
        meta: pydicom.Dataset, tag: MRITags, default_value: typing.Any = None
) -> typing.Any:
    """
    Get a corresponding value for a tag in a dicom metadata.

    :param meta: dicom file metadata
    :param tag: possible tag codes
    :param default_value: default value, if the tag is not found
    :return: the corresponding value stored by the tag
    """
    for tag_chain in tag.value:
        try:
            if type(tag_chain) in [tuple, list]:
                value = copy.deepcopy(meta)
                for chain_key in tag_chain:
                    value = value[chain_key]
            else:
                value = meta[tag_chain]
            return value.value
        except (KeyError, TypeError):
            ...
    return default_value


def set_tag_from_meta(meta: pydicom.Dataset, tag: MRITags, value: typing.Any) -> bool:
    """
    Set a corresponding value for a tag in a dicom metadata.

    :param meta: dicom file metadata
    :param tag: possible tag codes
    :param value: value to be set
    :return: ``True`` if the change was successful, ``False`` otherwise
    """
    for tag_chain in tag.value:
        try:
            if type(tag_chain) in [tuple, list]:
                tag_value = copy.deepcopy(meta)
                for chain_key in tag_chain:
                    tag_value = tag_value[chain_key]
            else:
                tag_value = meta[tag_chain]
            tag_value.value = value
            return True
        except KeyError:
            ...
    return False


args = parser.parse_args()

if len(args.file) != 0 and args.dir is not None:
    raise argparse.ArgumentError(args.dir, "Either file or dir should be specified.")

DATA_ROOT = args.dir or Path(get_env("DATA_ROOT"))
OUTPUT_DIR = args.out_dir or (DATA_ROOT / ".." / "ADNI-structured")
OUTPUT_DIR.mkdir(exist_ok=True)

if len(args.file) == 0:
    dirs = sorted(set([_.parent for _ in DATA_ROOT.rglob("*/Resting_State_fMRI/*/*/*.dcm")]))
else:
    dirs = args.file

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

DATA_FILE = DATA_ROOT / args.out_file

if DATA_FILE.exists():
    data = pd.read_csv(DATA_FILE, index_col=0, header=0)
else:
    data = pd.DataFrame(
        index=pd.Index(
            [],
            name="index"
        ),
        columns=[
            "patient_id", "session_time", "session_name",
            "image_type", "magnetic_field_strength", "patient_position",
            "patient_sex", "patient_age", "n_volumes", "n_slices",
            "record_time_per_volume", "record_time_per_slice",
            "image_height", "image_width",
            "slice_thickness", "x_pixel_spacing", "y_pixel_spacing",
            "direction_x", "direction_y", "direction_z",
            "TR_original", "TR", "contrast",
        ],
    )
    data.to_csv(DATA_FILE)

logger.info(f"Number of image groups in ADNI dataset: {len(dirs)}")
for index, session_name_dir in enumerate(dirs, start=1):
    iteration = f"{index} / {len(dirs)}"
    session_time_dir = session_name_dir.parent
    image_type_dir = session_time_dir.parent
    patient_dir = image_type_dir.parent
    patient_id = patient_dir.name

    session_time = datetime.datetime.strptime(
        session_time_dir.name, "%Y-%m-%d_%H_%M_%S.%f"
    )
    session_date = session_time.strftime("%Y_%m_%d")

    logger.info(f"{iteration} {session_name_dir}")
    logger.info(f"{iteration} session time = {session_time_dir.name}")

    subdir = ""
    for preprocess_dir in ["FunRaw", "T1Raw"]:
        time_format = "%H%M%S"
        logger.info(f"{iteration} {time_format = }")
        logger.info(f"{iteration} {preprocess_dir = }")
        logger.info(f"{iteration} patient id = {patient_id}")

        fmri_scans = [_ for _ in session_name_dir.glob("*.dcm")]
        if len(fmri_scans) == 0:
            logger.info(f"{iteration}\t\tSKIPPED {session_name_dir}")
            # Do not process anatomical image if there are no scans in fMRI folder
            break
        logger.info(f"{iteration}\t\tnumber of images = {len(fmri_scans)}")
        # read the meta information of the first image
        images_meta = fmri_scans[:1] | pipe.map(pydicom.dcmread) | pipe.to_list()

        n_volumes = int(get_tag_from_meta(
            images_meta[0], MRITags.number_of_volumes, 1
        ))
        n_slices = int(get_tag_from_meta(
            images_meta[0],
            MRITags.number_of_slices_per_volume,
            len(fmri_scans) // n_volumes
        ))

        record_time = get_tag_from_meta(
            images_meta[0], MRITags.acquisition_duration, 0.0
        )
        record_time_per_volume = record_time / n_volumes
        record_time_per_slice = record_time_per_volume / n_slices

        image_position = get_tag_from_meta(images_meta[0], MRITags.image_position,
                                           [0, 0, 0])
        positions = [
            "left-right" if image_position[0] > 0 else "right-left",
            "front-back" if image_position[1] > 0 else "back-front",
            "down-top" if image_position[2] > 0 else "top-down",
        ]
        tr_original = get_tag_from_meta(images_meta[0], MRITags.repetition_time)
        div = 1
        if tr_original > 1000:
            div = 1000
        tr = int(np.round(tr_original / div, 0))
        if subdir == "":
            subdir = f"{positions[2]}_{tr}"

        output_dir = (OUTPUT_DIR /
                      subdir /
                      preprocess_dir /
                      f"{patient_id}-{session_date}")

        acquisition_type = get_tag_from_meta(
            images_meta[0], MRITags.acquisition_type
        )
        logger.info(f"{iteration}\t\tacquisition type: {acquisition_type}")

        if not output_dir.exists():
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(session_name_dir, output_dir)

        found = data[(data["patient_id"] == patient_id) & (data["session_time"] == session_date) &
                     (data["session_name"] == session_name_dir.name)]
        if len(found) == 1:
            index = found.index[0]
        if len(found) > 1:
            raise IndexError(f"Index is not unique {found}")
        else:
            index = len(data)

        data.loc[index] = [
            patient_id, session_date, session_name_dir.name,
            get_tag_from_meta(images_meta[0], MRITags.protocol_name, ""),
            get_tag_from_meta(images_meta[0], MRITags.magnetic_filed_strength, ""),
            get_tag_from_meta(images_meta[0], MRITags.patient_position, ""),
            get_tag_from_meta(images_meta[0], MRITags.patient_sex, ""),
            get_tag_from_meta(images_meta[0], MRITags.patient_age, ""),
            n_volumes,
            n_slices,
            record_time_per_volume,
            record_time_per_slice,
            get_tag_from_meta(
                images_meta[0], MRITags.row, images_meta[0].pixel_array.shape[-2]
            ),
            get_tag_from_meta(
                images_meta[0],
                MRITags.column,
                images_meta[0].pixel_array.shape[-1]
            ),
            get_tag_from_meta(images_meta[0], MRITags.slice_thickness),
            *get_tag_from_meta(images_meta[0], MRITags.pixel_spacing),
            *positions,
            tr_original,
            tr,
            get_tag_from_meta(images_meta[0], MRITags.acquisition_contrast),
        ]
        data.to_csv(DATA_FILE)
        del images_meta
        del fmri_scans
        # change dir to point to anatomical image dir on the same date
        session_name_dir = [d for d in patient_dir.rglob(f"MPRAGE/{session_time.strftime('%Y-%m-%d')}*/*")][0]


data.to_csv(DATA_FILE)
