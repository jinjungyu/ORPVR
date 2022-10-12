import cv2
import os
import numpy as np
import json
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm

import torch

from util.option_relocate import args, Relocator
            
def main(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modes = ['original','offset','dynamic']
    modelname = os.path.basename(args.src)
    clip = os.path.basename(os.path.dirname(args.src))
    
    bimgdir = args.src
    imgdir = os.path.join('dataset',clip,'images')
    objdir = os.path.join('dataset',clip,'objects')
    args.resultdir = os.path.join(args.dstdir,clip,modelname,modes[args.mode])
    os.makedirs(args.resultdir,exist_ok=True)
    
    flist = []
    for ext in ['*.png','*.jpg']:
        flist.extend(glob(os.path.join(bimgdir,ext)))
    flist.sort()
    
    ilist = []
    for ext in ['*.png','*.jpg']:
        ilist.extend(glob(os.path.join(imgdir,ext)))
    ilist.sort()
    
    olist = glob(os.path.join(objdir,'*.json'))
    olist.sort()

    bimg = cv2.imread(flist[0],cv2.IMREAD_COLOR)
    args.h,args.w,_ = bimg.shape
    args.new_w = int(np.ceil(args.h * 16 / 9)) # 640 -> 854
    
    relocator = Relocator(args)
    
    print("Start Relocating....")
    for i in tqdm(range(len(flist))):
        # Inpainted Background
        fname = os.path.basename(flist[i])
        bimg = cv2.imread(flist[i],cv2.IMREAD_COLOR)
        bimg = cv2.resize(bimg,dsize=(args.new_w,args.h),interpolation=cv2.INTER_CUBIC)
        # Original Image
        img = cv2.imread(ilist[i],cv2.IMREAD_COLOR)
        with open(olist[i],"r") as f:
            objects = json.load(f)
        # Boxes and Objects
        # print(fname)
        frames = relocator(bimg,img,objects)
        
        if not isinstance(frames,list):
            cv2.imwrite(os.path.join(args.resultdir,fname), frames)
        elif len(frames) == 1:
            cv2.imwrite(os.path.join(args.resultdir,fname), frames[0])
        else:
            for i in range(len(frames)):
                newfname = fname.replace('.',f'_{i}.')
                cv2.imwrite(os.path.join(args.resultdir,newfname), frames[i])
    print(f"Object Relocated Images are stored in {args.resultdir}")
    print("Complete")

if __name__ == '__main__':
    main(args)
    
    