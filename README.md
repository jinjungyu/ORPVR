# Video Resolution Conversion - Using Instance Segmentation  

## Dependencies and Installation

**Step 1.** Clone Repository
```bash
git clone https://github.com/realJun9u/ORPVR.git
```

**Step 2.** Create Virtual Environment

*Choice 1. Anaconda*
**1.** Conda Environment
```bash
conda create --name $ENVNAME python=3.8 -y
conda activate $ENVNAME
```

**2.** Install Pytorch Following [Instructions](https://pytorch.org/get-started/locally/)
```bash
conda install pytorch torchvision torchaudio cudatoolkit=$CUDAVERSION -c pytorch -y
```

*Choice 2. Docker*
**1.** Install nvidia-docker [Instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and Create Python Contatainer
```bash
docker pull python:3.8
docker run -it --gpus '"device={device id}"' python:3.8 /bin/bash
```

**2.** Install Pytorch in Container. refer to [Instructions](https://pytorch.org/get-started/previous-versions)
```bash
# example of Pytorch 1.11.0v for Cuda 11.3v
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

**Step 3.** Install dependencies and Prepare Pretrained Models

1. Instance Segmentation [mmdetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md/#installation) following the pipeline to install
```bash
pip install -U openmim
mim install mmcv-full

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
mkdir checkpoints
pip install -v -e .
```
Download [mask2former swin-s backbone model checkpoint](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md/#installation) at mmdetection/checkpoints/
  
2. Video Inpainting [E2FGVI-HQ](https://github.com/MCG-NKU/E2FGVI)
```bash
git clone https://github.com/MCG-NKU/E2FGVI.git
```
Download [E2FGVI-HQ model checkpoint](https://github.com/MCG-NKU/E2FGVI#prepare-pretrained-models) at E2FGVI/release_model/

+) If you want to use image inpainting model
```bash
git clone https://github.com/hyunobae/AOT-GAN-for-Inpainting.git
```

## Quick Use
End-to-End Use
```
bash en2end.sh
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
