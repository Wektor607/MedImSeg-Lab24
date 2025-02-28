# Laboratory Work

## Overview
This work was conducted based on the following paper:
[Active Domain Adaptation via Clustering Uncertainty-Weighted Embeddings (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Prabhu_Active_Domain_Adaptation_via_Clustering_Uncertainty-Weighted_Embeddings_ICCV_2021_paper.pdf)

## Objective
The main goal of this study was to evaluate whether the proposed approach is applicable to the segmentation task.

## Implementation Details
The core implementation is located in the **src** directory. A complete description of all parameters is provided in the `run.py` file.

## Pretrained UNet Model
A pretrained **UNet** model can be downloaded from the following link:  
[UNet Model](https://drive.google.com/drive/folders/1KcH5gf4lDdV092uR3ryPlISmuwr1uJPU?usp=sharing)

## Running the Code

### Step 1: Setting Up the Environment
To create a virtual environment and install dependencies, run the following commands:

```bash
python3 -m venv CLUE 
source CLUE/bin/activate 
pip install -r requirements.txt
```

### Step 2: Activating the Environment
Before running the code, activate the virtual environment:

```bash
source ./CLUE/bin/activate
```

### Step 3: Running the Script
Execute the following command to run the code with default parameters:

```bash
python3 run.py --train False --num_clusters 20 --clue_softmax_t 0.1 \
--adapt_num_epochs 10 --device cuda:0 --uncertainty CrossEntropy \
--kernel_size 5 --stride 2 --target_size 4096
```

## Notes
- Ensure that the correct CUDA device is specified in the `--device` parameter.
- Modify the parameters as needed to adapt to different segmentation tasks.
