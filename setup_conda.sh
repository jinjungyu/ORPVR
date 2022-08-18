#!/bin/bash
echo "This project is dependent on the mmdetection project."
echo "Do you want to create a new Anaconda environment and install Pytorch?"

while true; do
    read -p "If you select No, install mmdetection in your environment [y/n]: " yn
    case $yn in
        [Yy]* ) conda create --name env_vid python=3.8 -y ; \
                source activate env_vid ; \
                read -p "Your Cuda Version [ex)10.2 or 11.3]: " cv
                conda install pytorch torchvision torchaudio cudatoolkit=$cv -c pytorch -y
                break;;
        [Nn]* ) break;;
        * ) echo "Please answer y or n";;
    esac
done

pip install -U openmim
mim install mmcv-full

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
mkdir checkpoints
pip install -v -e .

# 3. Segmentation Model checkpoint and config file Download

wget -c https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth \
-O checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth
