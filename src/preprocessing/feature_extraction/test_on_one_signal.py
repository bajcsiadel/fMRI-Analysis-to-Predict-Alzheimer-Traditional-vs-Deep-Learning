from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing.feature_extraction.features.high_order.associated import \
    static_associated_high_order_fc
from preprocessing.feature_extraction.features.high_order.topographical import \
    static_topographical_high_order_fc
from preprocessing.feature_extraction.features.low_order import static_low_order_fc
from utils.environment import get_env

dir_path = Path(
    get_env("DATA_ROOT"), "ADNI", "Signals", "slow5_ROISignals_FunImgARCWSF"
)
# patient_id = "002_S_0295-2011_06_02"  # NC
patient_id = "002_S_0729-2011_08_16"  # MCI
# patient_id = ""  # eMCI
# patient_id = ""  # lMCI
# patient_id = "006_S_4546-2012_03_05"  # AD

roi_correlation_filename = f"ROICorrelation_{patient_id}.txt"
roi_fisher_correlation_filename = f"ROICorrelation_FisherZ_{patient_id}.txt"
signal_filename = f"ROISignals_{patient_id}.txt"

plt.figure(figsize=(10, 5))
correlation = np.loadtxt(dir_path / roi_correlation_filename)
print(f"{correlation.shape = }")
print(f"{correlation.min() = }")
print(f"{correlation.max() = }")
plt.imshow(correlation, cmap="RdYlBu_r")
plt.axis("off")
plt.title("ROI correlation")
plt.show()

plt.figure(figsize=(10, 5))
fisher_correlation = np.loadtxt(dir_path / roi_fisher_correlation_filename)
print(f"{fisher_correlation.shape = }")
print(f"{fisher_correlation.min() = }")
print(f"{fisher_correlation.max() = }")
plt.imshow(fisher_correlation, cmap="RdYlBu_r")
plt.axis("off")
plt.title("ROI FisherZ correlation")
plt.show()

signals = np.loadtxt(dir_path / signal_filename)

plt.figure(figsize=(10, 5))
for signal in signals.T:
    plt.plot(signal)
plt.title("ROI signals")
plt.show()

C = static_low_order_fc(signals)
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
sns.heatmap(C[0], cmap="RdYlBu_r", ax=ax[0])
ax[0].axis("off")
ax[0].set_title("Low-order FC")

H = static_topographical_high_order_fc(C=C)
sns.heatmap(H[0], cmap="RdYlBu_r", ax=ax[1])
ax[1].axis("off")
ax[1].set_title("Topographical how-order FC")

A = static_associated_high_order_fc(C=C, H=H)
sns.heatmap(A[0], cmap="RdYlBu_r", ax=ax[2])
ax[2].axis("off")
ax[2].set_title("Associated how-order FC")
plt.show()
