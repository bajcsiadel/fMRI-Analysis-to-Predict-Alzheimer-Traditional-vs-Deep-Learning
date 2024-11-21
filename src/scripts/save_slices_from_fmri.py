"""
Save given slices from a 4D fMRI image.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nilearn import image


parser = argparse.ArgumentParser(__file__, description="Save given slices from a 4D fMRI image.")
parser.add_argument("fMRI", help="Path to the 4D fMRI image.", type=Path)
parser.add_argument("--slices", "-s", nargs="+", type=int, help="Slices to save.", required=True)
parser.add_argument("--starting-time", "-st", type=int, help="Starting time.", default=11)
parser.add_argument("--ending-time", "-et", type=int, help="Starting time.", default=100)
parser.add_argument("--gap", "-g", type=int, help="Gap between the time points.", default=11)
parser.add_argument("--output", "-o", help="Output directory.", required=True, type=Path)

args = parser.parse_args()

img = image.load_img(args.fMRI).get_fdata()
time_points = img.shape[-1]

if args.starting_time < 0 or args.starting_time > time_points:
    raise argparse.ArgumentError("starting_time", "Starting time should be greater than or equal to 0 and ending time should be less than the total time points.")
if args.ending_time < 0 or args.ending_time > time_points:
    raise argparse.ArgumentError("ending_time", "Ending time should be greater than or equal to 0 and ending time should be less than the total time points.")
if args.starting_time >= args.ending_time:
    raise argparse.ArgumentError("ending_time", "Ending time should be greater than the starting time.")

for slice in args.slices:
    output_dir = args.output / f"slice-{slice}"
    output_dir.mkdir(exist_ok=True, parents=True)

    for t in range(args.starting_time, args.ending_time, args.gap):
        current_timepoint = img[:, :, slice, t]
        print(current_timepoint.shape)
        fig, ax = plt.subplots(1, 1)
        current_timepoint = np.rot90(current_timepoint)
        ax.imshow(current_timepoint, cmap="gray")
        ax.axis("off")
        # plt.savefig(output_dir / f"{args.fMRI.name}_t{t}.png", dpi=300, bbox_inches="tight", pad_inches=0)
        plt.savefig(output_dir / f"ADNI_006_S_4449_t{t}.png", dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()