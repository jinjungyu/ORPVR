# Video Resolution Conversion - Using Instance Segmentation  

## Prerequisites (selection)
**Step 1.** Create Virtual Environment  
step
```bash
conda create --name $ENVNAME python=3.8 -y
conda activate $ENVNAME
```
**Step 2.** Install Pytorch Following [Instructions](https://pytorch.org/get-started/locally/)
```bash
conda install pytorch torchvision torchaudio cudatoolkit=$CUDAVERSION -c pytorch -y
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
