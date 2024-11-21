# fMRI Analysis for Alzheimer's Disease Detection: Traditional vs. Deep Learning Models

The current project aims to classify Alzheimer's disease using resting-state function Magnetic Resonance Images (rs-fMRI). We include both traditional machine learning and deep learning to compare their performance.

The steps of the systems are the following:
1. Preprocessing (using DPARSF). Details in [`src/preprocessing/README.md`](src/preprocessing/README.md)
2. Feature extraction: we use brain functional connectivity as detailed in [`src/preprocessing/feature_extraction/README.md`](src/preprocessing/feature_extraction/README.md)
3. Classification: using tradition machine learning (DT, RF, KNN, and SVM) and deep learning (AlexNet, ResNet-18, and ResNet-50)

## Setup

1. Install [Poetry](https://python-poetry.org/docs/)
2. Install environment
    ```shell
    poetry install
    ```
3. Activate environment
    ```shell
    poetry shell
    ```
4. Run the code
    ```shell
    python src/scripts/classify_fmri.py
    ```
   
## Configuration

Create a copy of the `.env.example`, name it as `.env`, and complete the value of the variables. This file contains the global settings of the project.

In this project [Hydra](https://hydra.cc/docs/intro/) is used to configure the scrips. The configuration files are located in `conf`. The name of the configuration directory can be changed in `.env` by setting the `CONFIGURATIONS_DIR_NAME` variable.
