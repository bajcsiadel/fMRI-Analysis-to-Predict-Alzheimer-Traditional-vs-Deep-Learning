# Preprocessing

The current folder contains the scripts responsible for the preprocessing of the
data described in [^Sethuraman-2023], section 3.2.

Before jumping into the preprocessing we need to convert and restructure the data.
ADNI contains Dicom images while SPM can work with Nifti.

First, call
```shell
python group_scans.py --out-dir <NIFTI_DIR>
```
then
```shell
python restructure.py --dir <NIFTI_DIR> --out-dir <STRUCTURED_DIR>
```

Now we can start the actual preprocessing.

In [^Sethuraman-2023], the authors propose the use of SPM 12 in the preprocessing.
However, DPARSF [^DPARSF] (Data Processing Assistant for rsMRI) is built on SPM12.
Therefore, we will use only DPARSF.

At the beginning of the preprocessing the brain tissue is segmented into three classes: Gray Matter (GM), White Matter (WM) 
and Cerebrospinal Fluid (CSF).

The following steps are performed:
1. **Removing the initial ten volumes**: usually the initial volumes are noisy,
therefore they are removed.
2. **Slice timing**: it takes time to record the full volume of the brain,
therefore usually slice timing is applied to provide uniformity in time 
variation. The **middle** slice is selected as the reference slice.
3. **Realignment**: the scans are realigned to remove noise cause by the motion
of the patient, with regard to the reference (middle) slice.
4. **Co-registration**: aligning the average functional image (defined in the
previous step) with the structural image using landmark-based registration 
technique.
5. **Segmentation**: the brain tissue is segmented into six classes
   * gray matter,
   * white matter, 
   * cerebrospinal fluid, 
   * bones,
   * soft tissue and
   * air/background
   
   We are interested in the first three.
6. **Normalization**: convert images to MNI (Montreal Neurological Institute) space (
voxel size <code>3 x 3 x 3 mm<sup>3</sup></code>)
7. **Smoothing**: Gaussian kernel is applied to further reduce noise.

Finally, the low frequency bands are defined:

* **slow4**: 0.027 Hz - 0.08  Hz
* **slow5**: 0.01  Hz - 0.027 Hz
* **full-band**: 0.01 Hz - 0.08 Hz

[^DPARSF]: Chao-Gan Y, Yu-Feng Z. DPARSF: A MATLAB Toolbox for "Pipeline" Data
  Analysis of Resting-State fMRI. Front Syst Neurosci. 2010;4:13. Published 2010
  May 14. [doi:10.3389/fnsys.2010.00013](https://doi.org/10.3389/fnsys.2010.00013)
[^Sethuraman-2023]: Sethuraman SK, Malaiyappan N, Ramalingam R, Basheer S, 
  Rashid M, Ahmad N. Predicting Alzheimer’s Disease Using Deep Neuro-Functional
  Networks with Resting-State fMRI. Electronics. 2023; 12(4):1031.
  [doi:10.3390/electronics12041031](https://doi.org/10.3390/electronics12041031)
[^SPM12]: [SPM12](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/)