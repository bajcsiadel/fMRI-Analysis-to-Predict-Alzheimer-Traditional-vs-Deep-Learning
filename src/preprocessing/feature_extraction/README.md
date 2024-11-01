# Feature extraction

Call the following command to extract the features from the ROI signals:
```shell
python preprocessing/feature_extraction/run.py 
```

From the generated ROI signals functional connectivity (FC) features are extracted
based on [^Zhang-2017]. Three types of FC are extracted:
1. traditional low-order FC
2. topological high-order FC
3. higher-level associated FC

Each feature has two versions:
1. *dynamic*: in [^Zhang-2017] `window_length = 70` and `step = 1` were used
2. *static* (a static network is an extreme case where window length is maximized
to the entire timescale)

## Traditional low-order FC

Computed using function `preprocessing.feature_extraction.features.low_order.
traditional.dynamic_low_order_fc`. With sliding window approach, an rs-fMRI 
time series can be segmented into multiple sub-series. In particular, 
$K = \frac{P - L}{S} + 1$ sub-series can be generated from an rs-fMRI time
series with $P$ time points, where $L$ and $S$ are the window length and step
size, respectively. The correlation of the resulting sub-series is defined by
using Pearson's correlation.

## Topographical high-order FC

Computed using function `preprocessing.feature_extraction.features.high_order.
topographical.dynamic_topographical_high_order_fc`. Low-order FC network on 
the $k^{th}$ time sub-series presents to connectivity of the ROIs. The $i^{th}$
row represents the connectivity of $ROI_i$ to all other ROIs. Therefore, we
regard each row as a “sub-network” between node $i$ and other regions. Then,
a high-order FC network can be constructed by calculating the FC between
every pair of the low-order sub-networks.

## Associated high-order FC

Computed using function `preprocessing.feature_extraction.features.high_order.
associated.dynamic_associated_high_order_fc`. The associated high-order FC 
network characterizes the inter-level interactions between the low-order
sub-networks (result of Traditional low-order FC) and high-order sub-networks
(result of Topographical high-order FC).


[^Zhang-2017] Zhang, Y., Zhang, H., Chen, X., Lee, S.-W., & Shen, D. (2017).
    Hybrid High-order Functional Connectivity Networks Using Resting-state
    Functional MRI for Mild Cognitive Impairment Diagnosis. Scientific Reports,
    7(1), 6530. [doi:10.1038/s41598-017-06509-0](https://doi.org/10.1038/s41598-017-06509-0)