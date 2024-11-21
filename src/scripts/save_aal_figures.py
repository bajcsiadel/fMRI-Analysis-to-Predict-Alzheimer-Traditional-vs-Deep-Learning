from pathlib import Path

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from nilearn import datasets
from nilearn import image
from nilearn import plotting
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker

my_atlas_file = Path("C:\\Users\\HP\\Downloads\\FCROI_1_006_S_4449-2013_03_12.nii")
fMRI = Path("C:\\Users\\HP\\Downloads\\sFiltered_4DVolume.nii")

atlas = datasets.fetch_atlas_aal()
# Loading atlas image stored in 'maps'
atlas_filename = atlas["maps"]
print(f"Atlas filename: {atlas_filename}")
# Loading atlas data stored in 'labels'
labels = atlas["labels"]

# Loading the functional datasets
data = image.load_img(fMRI)
# data = datasets.fetch_development_fmri(n_subjects=1)
img = image.load_img(data)
print(f"Data: {data}")
print(f"Shape of the 4D image: {img.shape}")

atlas_data = image.load_img(atlas_filename)
# atlas_data = image.resample_to_img(atlas_data, img)
print(f"Shape of the atlas: {atlas_data.shape}")
print(np.unique(atlas_data.get_fdata()))
my_atlas = image.load_img(my_atlas_file)
print(my_atlas.shape)

# plotting.plot_glass_brain(atlas_data, cmap="tab20", alpha=0.8)
plotting.plot_roi(my_atlas, cmap="tab20", alpha=0.8)
plt.savefig("aal-applied.pdf", dpi=300, bbox_inches="tight")
exit()

masker = NiftiLabelsMasker(
    labels_img=atlas_data,
    standardize="zscore_sample",
)
time_series = masker.fit_transform(data)
time_series_df = pd.DataFrame(time_series, columns=labels)
time_series_df.to_csv(fMRI.with_name(fMRI.name + "-signal").with_suffix(".csv"))

print(time_series.shape)

correlation_measure = ConnectivityMeasure(
    kind="correlation",
    standardize="zscore_sample",
)
correlation_matrix = correlation_measure.fit_transform([time_series])[0]
pd.DataFrame(correlation_matrix, columns=labels, index=labels).to_csv(fMRI.with_name(fMRI.name + "-FC").with_suffix(".csv"))

# Display the correlation matrix
plt.plot(time_series_df['Precentral_L'])
plt.title('Time Series for Precentral_L Region')
plt.xlabel('Time points')
plt.ylabel('Signal')
plt.show()

# Mask out the major diagonal
np.fill_diagonal(correlation_matrix, 0)
plotting.plot_matrix(
    correlation_matrix, labels=labels, colorbar=True, vmax=0.8, vmin=-0.8
)
plotting.show()

coords = plotting.find_parcellation_cut_coords(labels_img=atlas_data)

# We threshold to keep only the 20% of edges with the highest value
# because the graph is very dense
plotting.plot_connectome(
    correlation_matrix, coords, edge_threshold="80%", colorbar=True, edge_vmin=0.0, edge_vmax=1.0
)

plotting.show()
