import argparse
from pathlib import Path

from utils.environment import get_env


def define_parser():
    parser = argparse.ArgumentParser(__file__, "Extract features from ROI signals")
    parser.add_argument(
        "-d", "--dir",
        dest="dir",
        type=Path,
        help="Directory containing the ROI signals",
    )
    parser.add_argument(
        "-o", "--out-dir",
        dest="out_dir",
        type=Path,
        help="Output directory",
    )
    parser.add_argument(
        "-e", "--regex",
        dest="regex",
        type=str,
        default=r"ROISignals_*.npz",
        help="Regular expression to match ROI signal files",
    )
    features_group = parser.add_argument_group(
        "Features", "Select the features to disable",
    )
    features_group.add_argument(
        "--disable-dynamic-low-order",
        dest="disable_dynamic_low_order",
        action="store_true",
        help="Disable dynamic low-order features"
    )
    features_group.add_argument(
        "--disable-static-low-order",
        dest="disable_static_low_order",
        action="store_true",
        help="Disable static low-order features"
    )
    features_group.add_argument(
        "--disable-dynamic-topographical-high-order",
        dest="disable_dynamic_topographical_high_order",
        action="store_true",
        help="Disable dynamic topographical high-order features"
    )
    features_group.add_argument(
        "--disable-static-topographical-high-order",
        dest="disable_static_topographical_high_order",
        action="store_true",
        help="Disable static topographical high-order features"
    )
    features_group.add_argument(
        "--disable-dynamic-associated-high-order",
        dest="disable_dynamic_associated_high_order",
        action="store_true",
        help="Disable dynamic associated high-order features"
    )
    features_group.add_argument(
        "--disable-static-associated-high-order",
        dest="disable_static_associated_high_order",
        action="store_true",
        help="Disable static associated high-order features"
    )

    result_dirs = parser.add_argument_group(
        "Result directories",
        "Name/location of the directories where the features are saved",
    )
    result_dirs.add_argument(
        "--dynamic-low-order-dir",
        dest="dynamic_low_order_dir",
        type=Path,
        default="DN_L",
        help="Directory where dynamic low-order features are saved"
    )
    result_dirs.add_argument(
        "--static-low-order-dir",
        dest="static_low_order_dir",
        type=Path,
        default="SN_L",
        help="Directory where static low-order features are saved"
    )
    result_dirs.add_argument(
        "--dynamic-topographical-high-order-dir",
        dest="dynamic_topographical_high_order_dir",
        type=Path,
        default="DN_H",
        help="Directory where dynamic topographical high-order features are saved"
    )
    result_dirs.add_argument(
        "--static-topographical-high-order-dir",
        dest="static_topographical_high_order_dir",
        type=Path,
        default="SN_H",
        help="Directory where static topographical high-order features are saved"
    )
    result_dirs.add_argument(
        "--dynamic-associated-high-order-dir",
        dest="dynamic_associated_high_order_dir",
        type=Path,
        default="DN_A",
        help="Directory where dynamic associated high-order features are saved"
    )
    result_dirs.add_argument(
        "--static-associated-high-order-dir",
        dest="static_associated_high_order_dir",
        type=Path,
        default="SN_A",
        help="Directory where static associated high-order features are saved"
    )

    feature_parameters_group = parser.add_argument_group(
        "Feature parameters", "Parameters for the features"
    )
    feature_parameters_group.add_argument(
        "--window-length",
        dest="window_length",
        type=int,
        default=70,
        help="Window length for the sliding window"
    )
    feature_parameters_group.add_argument(
        "--stride",
        dest="stride",
        type=int,
        default=1,
        help="Stride for the sliding window"
    )

    return parser


def parse_args():
    parser = define_parser()
    args = parser.parse_args()

    args.dir = args.dir or Path(get_env("DATA_ROOT"))

    if args.out_dir is None:
        args.out_dir = args.dir / "ADNI" / "Features"

    for param_name in (
            "dynamic_low_order_dir",
            "static_low_order_dir",
            "dynamic_topographical_high_order_dir",
            "static_topographical_high_order_dir",
            "dynamic_associated_high_order_dir",
            "static_associated_high_order_dir"
    ):
        result_dir = getattr(args, param_name)
        if not result_dir.is_absolute():
            setattr(args, param_name, args.out_dir / result_dir)
            getattr(args, param_name).mkdir(parents=True, exist_ok=True)

    return args
