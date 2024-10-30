# HistoDAVAE: Dependency-aware Deep Generative Model for Inferring Super-resolved Spatial Transcriptomics via Histology Images
#### Yuqi Chen†, Peng Jiang†, Yinbo Liu, Feng Yang, Juan Liu and Tian Tian
HistoDAVAE is a Variational Autoencoder (VAE)-based deep generative model to predict super-resolution gene expression from histology images. HistoDAVAE employs a combined embedding of Gaussian process (GP) prior and Gaussian prior to explicitly model spatial correlations among spots. 

## Directory structure

```
.
├── data                    # Data files
├── Preprocess_her2.ipynb   # Tutorial for preprocessing her2st dataset
├── gen_gene_list.py        # Code for gene list generation 
├── gen_mask.py             # Code to generate mask for HE images
├── gen_new_image.py        # Code to generate cropped mask for HE images (background removal)
├── image_preprocess.py     # Code to rescale and adjust mask and HE images
├── HistoDAVAE.py           # Model structure
├── run_HistoDAVAE.py       # Main script for model training and test
├── SVGP.py                 # Code for Sparse Variational Gaussian Process
├── kernel.py               # Kernel function for Gaussian Process
├── I_PID.py                # Code for incremental PID algorithm
├── VAE_utils.py            # Submodules and functions for model construction
├── utils.py                # General utility functions or helper functions
├── requirements.txt        # Reproducible Python environment via pip
└── README.md
```
For data Structure of her2st dataset, please refer to: https://github.com/almaan/her2st

## System environment
Required package:
- Python >= 3.9
- PyTorch >= 2.0
- scanpy >= 1.8

To install dependencies, users can use the below command:

```
pip install -r requirements.txt
```

## HistoDAVAE pipeline
Step 1：Convert spatial transcriptomics data to .h5ad format.  
Step 2：Generate gene lists for predition.  
Step 3：Generate initial mask for raw HE image.  
Step 4：Generate cropped HE and mask images along with the cropped coordinates (To remove most of the background area).  
Step 5：Adjust HE and mask images for training.  
Step 6：Train and test HistoDAVAE model.  


# Usage

Train HistoDAVAE model with:
```
python run_HistoDAVAE.py 
```

Test HistoDAVAE model with:
```
python run_HistoDAVAE.py --predict_only 
```

## References
her2st dataset: https://github.com/almaan/her2st
