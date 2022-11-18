# Video Resolution Conversion - Using Instance Segmentation  

## Prerequisites (selection)

### 1. Anaconda

**Step 1.** Create Virtual Environment  
```bash
conda create --name $ENVNAME python=3.8 -y
conda activate $ENVNAME
```
**Step 2.** Install Pytorch Following [Instructions](https://pytorch.org/get-started/locally/)
```bash
# example
conda install pytorch torchvision torchaudio cudatoolkit=$CUDAVERSION -c pytorch -y
```

### 2. Docker
**Step 1.** Install nvidia-docker [Instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

**Step 2.** Create Python Container
```bash
docker pull python:3.8
docker run -it --gpus '"device={device id}"' python:3.8 /bin/bash
```

**Step 3.** Install Pytorch in Container. refer to [Instructions](https://pytorch.org/get-started/previous-versions)
```bash
# example of Pytorch 1.11.0v for Cuda 11.3v
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Installation and Use  
**Step 1.** Install MMDetection
```bash
bash install.sh
```  
**Step 2.** Masking Rawdataset  
You can use a directory containing images or a single image as an input.  
The file path is always considered to be under rawdataset.  
```bash
# Directory
python setup_data.py $video/$scene#
```
```bash
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
