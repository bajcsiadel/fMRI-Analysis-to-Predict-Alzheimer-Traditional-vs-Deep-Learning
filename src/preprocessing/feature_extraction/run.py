"""
Extract features
================

Extracting features from the ROI signals.

Arguments
=========

-h, --help            show this help message and exit
  -d DIR, --dir DIR     Directory containing the ROI signals
  -o OUT_DIR, --out-dir OUT_DIR
                        Output directory
  -e REGEX, --regex REGEX
                        Regular expression to match ROI signal files

Features:
  Select the features to disable

  --disable-dynamic-low-order
                        Disable dynamic low-order features
  --disable-static-low-order
                        Disable static low-order features
  --disable-dynamic-topographical-high-order
                        Disable dynamic topographical high-order features
  --disable-static-topographical-high-order
                        Disable static topographical high-order features
  --disable-dynamic-associated-high-order
                        Disable dynamic associated high-order features
  --disable-static-associated-high-order
                        Disable static associated high-order features
  --dynamic-low-order-dir DYNAMIC_LOW_ORDER_DIR
                        Directory where dynamic low-order features are saved
  --static-low-order-dir STATIC_LOW_ORDER_DIR
                        Directory where static low-order features are saved
  --dynamic-topographical-high-order-dir DYNAMIC_TOPOGRAPHICAL_HIGH_ORDER_DIR
                        Directory where dynamic topographical high-order
                        features are saved
  --static-topographical-high-order-dir STATIC_TOPOGRAPHICAL_HIGH_ORDER_DIR
                        Directory where static topographical high-order
                        features are saved
  --dynamic-associated-high-order-dir DYNAMIC_ASSOCIATED_HIGH_ORDER_DIR
                        Directory where dynamic associated high-order features
                        are saved
  --static-associated-high-order-dir STATIC_ASSOCIATED_HIGH_ORDER_DIR
                        Directory where static associated high-order features
                        are saved

Result directories:
  Name/location of the directories where the features are saved

Feature parameters:
  Parameters for the features

  --window-length WINDOW_LENGTH
                        Window length for the sliding window
  --stride STRIDE       Stride for the sliding window
"""

from pathlib import Path

import numpy as np
from icecream import ic
from tqdm import tqdm

from preprocessing.feature_extraction.args import parse_args
from preprocessing.feature_extraction.features.high_order.associated import (
    dynamic_associated_high_order_fc,
    static_associated_high_order_fc,
)
from preprocessing.feature_extraction.features.high_order.topographical import (
    dynamic_topographical_high_order_fc,
    static_topographical_high_order_fc,
)
from preprocessing.feature_extraction.features.low_order import (
    dynamic_low_order_fc,
    static_low_order_fc,
)

args = parse_args()

signal_files = [_ for _ in args.dir.rglob(args.regex)]
for signal_file in tqdm(signal_files, desc="Extracting featured from ROI signals"):
    signal = np.load(signal_file)["signal"]

    out_dir = args.out_dir / signal_file.parent.name.split("_")[0]
    feature_file_name = "_".join(signal_file.name.split("_")[1:])

    results = {}
    for disabled, extract_feature_fn, kwargs, result_dir in [
        (
            args.disable_dynamic_low_order,
            dynamic_low_order_fc,
            {
                "signal": signal,
                "window_length": args.window_length,
                "stride": args.stride,
            },
            args.dynamic_low_order_dir,
        ),
        (
            args.disable_static_low_order,
            static_low_order_fc,
            {"signal": signal},
            args.static_low_order_dir,
        ),
        (
            args.disable_dynamic_topographical_high_order,
            dynamic_topographical_high_order_fc,
            {"C": "results[args.dynamic_low_order_dir]"},
            args.dynamic_topographical_high_order_dir,
        ),
        (
            args.disable_static_topographical_high_order,
            static_topographical_high_order_fc,
            {"C": "results[args.static_low_order_dir]"},
            args.static_topographical_high_order_dir,
        ),
        (
            args.disable_dynamic_associated_high_order,
            dynamic_associated_high_order_fc,
            {
                "C": "results[args.dynamic_low_order_dir]",
                "H": "results[args.dynamic_topographical_high_order_dir]",
            },
            args.dynamic_associated_high_order_dir,
        ),
        (
            args.disable_static_associated_high_order,
            static_associated_high_order_fc,
            {
                "C": "results[args.static_low_order_dir]",
                "H": "results[args.static_topographical_high_order_dir]",
            },
            args.static_associated_high_order_dir,
        ),
    ]:
        if not disabled:
            for k, v in kwargs.items():
                if type(v) is str:
                    kwargs[k] = eval(v)

            results[result_dir] = extract_feature_fn(**kwargs)

            if not (out_dir / result_dir).exists():
                (out_dir / result_dir).mkdir(parents=True)

            np.savez(
                out_dir / result_dir / feature_file_name, feature=results[result_dir]
            )
