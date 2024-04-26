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


parser = argparse.ArgumentParser(__file__, "Convert DICOM images to NIFTI")
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


def get_tag_from_meta(meta, tag, default_value=None):
    """
    Get a corresponding value for a tag in a dicom metadata.

    :param meta: dicom file metadata
    :type meta: pydicom.Dataset
    :param tag: possible tag codes
    :type tag: MRITags
    :param default_value: default value, if the tag is not found
    :type default_value: any
    :return: the corresponding value stored by the tag
    :rtype: any
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
        except KeyError:
            ...
    return default_value


def set_tag_from_meta(meta, tag, value):
    """
    Set a corresponding value for a tag in a dicom metadata.

    :param meta: dicom file metadata
    :type meta: pydicom.Dataset
    :param tag: possible tag codes
    :type tag: MRITags
    :param value: value to be set
    :type value: any
    :return: ``True`` if the change was successful, ``False`` otherwise
    :rtype: bool
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
OUTPUT_DIR = args.out_dir or (DATA_ROOT / ".." / "grouped")

if len(args.file) == 0:
    dirs = sorted(set([_.parent for _ in DATA_ROOT.rglob("*.dcm")]))
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

DATA_FILE = DATA_ROOT / "data.csv"
image_groups_file = DATA_ROOT / "image_groups.json"

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

if image_groups_file.exists():
    with image_groups_file.open("r") as fd:
        grouped_image_files = json.load(fd)
        copy_full = copy.deepcopy(grouped_image_files)
else:
    grouped_image_files = {}

logger.info(f"Number of image groups in ADNI dataset: {len(dirs)}")
for index, session_name_dir in enumerate(dirs, start=1):
    iteration = f"{index} / {len(dirs)}"
    session_time_dir = session_name_dir.parent
    image_type_dir = session_time_dir.parent
    patient_dir = image_type_dir.parent

    logger.info(f"{iteration} {session_name_dir}")
    logger.info(f"{iteration} session time = {session_time_dir.name}")

    time_format = "%H%M%S"
    image_type = "structural"
    preprocess_dir = "T1Img"
    if "Rest" in image_type_dir.name:
        image_type = "functional"
        preprocess_dir = "FunImg"
    logger.info(f"{iteration} {time_format = }")
    logger.info(f"{iteration} {image_type = }")
    logger.info(f"{iteration} patient id = {patient_dir.name}")

    grouped_image_files[patient_dir.name] = {
        session_time_dir.name: {}
    }
    copy_full[patient_dir.name] = {
        session_time_dir.name: {}
    }

    fmri_scans = [_ for _ in session_name_dir.iterdir()
                  if _.is_file() and _.suffix == ".dcm"]
    if len(fmri_scans) == 0:
        logger.info(f"{iteration}\t\tSKIPPED {session_name_dir}")
        continue
    logger.info(f"{iteration}\t\tnumber of images = {len(fmri_scans)}")
    images_meta = fmri_scans | pipe.map(pydicom.dcmread) | pipe.to_list()

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

    session_time = datetime.datetime.strptime(
        session_time_dir.name, "%Y-%m-%d_%H_%M_%S.%f"
    )

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

    output_dir = (OUTPUT_DIR /
                  f"{patient_dir.name}-"
                  f"{session_time.strftime('%Y_%m_%d')}" /
                  image_type)

    acquisition_type = get_tag_from_meta(
        images_meta[0], MRITags.acquisition_type
    )
    logger.info(f"{iteration}\t\tacquisition type: {acquisition_type}")

    grouped_scans = {}
    copy_current = {}
    if len(images_meta) == 1 and (acquisition_type.upper() == "3D" or n_slices > 1):
        logger.info(f"{iteration}\t\tn_volumes: {n_volumes}")
        logger.info(f"{iteration}\t\tn_slices: {n_slices}")
        n_frames = get_tag_from_meta(images_meta[0], MRITags.n_frames)
        for i in range(n_volumes):
            grouped_scans[i] = [copy.deepcopy(images_meta[0])]
            volume_slice = slice(i, n_frames, n_volumes)
            grouped_scans[i][0].PixelData = images_meta[0].pixel_array[volume_slice].tobytes()
            grouped_scans[i][0].PerFrameFunctionalGroupsSequence = grouped_scans[i][0].PerFrameFunctionalGroupsSequence[volume_slice]
            for frame in grouped_scans[i][0].PerFrameFunctionalGroupsSequence:
                frame[0x00209111][0][0x00209157].value = frame[0x00209111][0][0x00209157].value[:2]
                frame[0x00209111][0][0x00209128].value = 1
            set_tag_from_meta(grouped_scans[i][0], MRITags.n_frames, n_slices)
    else:
        # 2D acquisition
        grouped_patient_scans = {}
        # group scans by acquisition time
        for scan_index, image_meta in enumerate(images_meta):
            acquisition_time = datetime.datetime.strptime(
                get_tag_from_meta(
                    image_meta, MRITags.acquisition_time, "000000"
                ).split(".")[0],
                time_format
            )
            if acquisition_time not in grouped_scans:
                grouped_scans[acquisition_time] = []
                copy_current[acquisition_time.strftime("%H-%M-%S")] = []
                grouped_patient_scans[acquisition_time] = []
            grouped_scans[acquisition_time].append(image_meta)
            grouped_patient_scans[acquisition_time].append(
                fmri_scans[scan_index]
            )
            copy_current[acquisition_time.strftime("%H-%M-%S")].append(str(fmri_scans[scan_index]))
        # sort elements by key (acquisition time)
        grouped_scans = OrderedDict(
            sorted(grouped_scans.items(), key=lambda g: g[0])
        )
        grouped_patient_scans = OrderedDict(
            sorted(grouped_patient_scans.items(), key=lambda g: g[0])
        )
        logger.info(f"{iteration}\t\t\tn_volumes = {len(grouped_scans)}")

        grouped_image_files[patient_dir.name][session_time_dir.name][image_type] = grouped_patient_scans
        copy_full[patient_dir.name][session_time_dir.name][image_type] = copy_current

        # assert len(grouped_scans) == len(fmri_scans) // n_slices
        scans_per_record = grouped_scans.values() | pipe.map(
            len) | pipe.to_numpy()
        # assert np.all(scans_per_record == n_slices)

        logger.info(f"{iteration}\t\t\t{n_slices = }")

        logger.info(f"{iteration}\t\t\t{record_time_per_volume = }")
        timestamps = list(grouped_scans.keys())
        if image_type == "functional":
            mid = len(timestamps) // 2
            record_time_per_slice = (
                        timestamps[mid] - timestamps[mid - 1]).seconds

    if not output_dir.exists():
        start_all = time.time()
        output_dir.mkdir(parents=True, exist_ok=True)
        for record_index, scans in enumerate(grouped_scans.values(),
                                             start=1):
            start_convert_volume = time.time()
            output_file = (f"{patient_dir.name}-{session_name_dir.name}-"
                           f"{record_index:04}")
            output_path = output_dir / output_file
            dicom_array_to_nifti(scans, output_file=output_path)
            logger.info(f"{iteration}\t\t\t"
                        f"{time.time() - start_convert_volume:.2f}s "
                        f"NiFTI saved to {output_path}")
        logging.info(f"{iteration}\t\t\t{time.time() - start_all:.2f}s")
    del grouped_scans

    found = data[(data["patient_id"] == patient_dir.name) & (data["session_time"] == session_time_dir.name) &
                 (data["session_name"] == session_name_dir.name)]
    if len(found) == 1:
        index = found.index[0]
    if len(found) > 1:
        raise IndexError(f"Index is not unique {found}")
    else:
        index = len(data)

    data.loc[index] = [
        patient_dir.name, session_time_dir.name, session_name_dir.name,
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
    data.to_csv(DATA_FILE.with_name("test.csv"))
    del images_meta
    del fmri_scans
    with image_groups_file.open("w") as f:
        json.dump(copy_full, f, indent=4)

data.to_csv(DATA_FILE)

with image_groups_file.open("w") as f:
    json.dump(copy_full, f)
