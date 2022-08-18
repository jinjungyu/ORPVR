# Video Resolution Conversion - Using Instance Segmentation  

## How to Use  
0. clone repository  
```
git clone https://github.com/realJun9u/Vid_Conversion.git
```
1. Create Virtual Environment and Install MMDetection  
This project is dependent on the mmdetection project.  
You can choose whether to create a virtual environment and install Pytorch or proceed with the current environment.  

* Directory Structure  
.  
├── README.md  
├── dataset  
├── mmdetection - will be installed  
├── rawdataset  
├── setup_conda.sh  
└── setup_data.py  

```bash
bash setup_conda.sh
conda activate env_vid
```

2. Masking Rawdataset  
You can use a directory containing images or a single image as an input.  
* The file path is always considered to be under rawdataset.  

```bash
# Directory
python setup_data.py REDS_640x480/$scene#
# Single image
python setup_data.py man.jpg
``` 
  
### CUDA Version, cuDNN Version Check  
1. CUDA  
```bash  
nvcc -V
```
2. cuDNN  
```bash
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```
