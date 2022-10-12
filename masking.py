import cv2
import os
import numpy as np
import json
from tqdm import tqdm
from glob import glob

import torch

from mmdetection.mmdet.apis import inference_detector,init_detector

from util.option_masking import args,compute_intersect_area

target = 0
subtarget = [24,26,28,67]

def segmentation(args,model):
    result = inference_detector(model,args.img)
    img = cv2.imread(args.img,cv2.IMREAD_COLOR)

    size = args.h*args.w

    mask = np.zeros((args.h,args.w),dtype=np.uint8)
    objects = {'box':[],'coor':[]}

    for k in range(len(result[0][target])): # 각 사람 돌기
        score = result[0][target][k][-1]
        # score threshold 보다 낮은 박스는 무시
        if score < args.score_thr:
            continue
        coor = set()
        c1,r1,c2,r2 = map(int,result[0][target][k][:-1])
        temp = result[1][target][k]
        for i in range(r1,r2):
            for j in range(c1,c2):
                if temp[i][j]:
                    coor.add((i,j))
        # 사람 객체가 작으면 마스킹 X
        if len(coor) / size < args.area_thr:
            continue
        # 서브 카테고리 마스킹
        x1,y1,x2,y2 = c1,r1,c2,r2
        for subidx in subtarget:
            for l in range(len(result[0][subidx])):
                score = result[0][subidx][l][-1]
                # score threshold 보다 낮은 박스는 무시
                if score < args.score_thr:
                    continue
                nc1,nr1,nc2,nr2 = map(int,result[0][subidx][l][:-1])
                subtemp = result[1][subidx][l]
                # 서브 객체 박스가 사람 객체와 겹치면 포함
                # if (c1 <= nc1 <= c2 and r1 <= nr1 <= r2) or (c1 <= nc2 <= c2 and r1 <= nr2 <= r2):
                #     x1,y1,x2,y2 = min(x1,nc1), min(y1,nr1), max(x2,nc2), max(y2,nr2)
                #     for i in range(nr1,nr2):
                #         for j in range(nc1,nc2):
                #             if subtemp[i][j]:
                #                 coor.add((i,j))
                # 2022/09/20 : 서브 카테고리 박스가 30% 이상 겹쳐야 포함으로 결정
                nsize = (nr2-nr1) * (nc2-nc1)
                interarea = compute_intersect_area([c1,r1,c2,r2],[nc1,nr1,nc2,nr2])
                if interarea / nsize > 0.3:
                    x1,y1,x2,y2 = min(x1,nc1), min(y1,nr1), max(x2,nc2), max(y2,nr2)
                    for i in range(nr1,nr2):
                        for j in range(nc1,nc2):
                            if subtemp[i][j]:
                                coor.add((i,j))
        for i,j in coor:
            mask[i][j] = 255
        
        objects['box'].append([x1,y1,x2,y2])
        objects['coor'].append(sorted(list(coor)))

    cv2.imwrite(os.path.join(args.imgdir,args.fname+'.'+args.ext), img)
    cv2.imwrite(os.path.join(args.maskdir,args.fname+'.png'), mask)
    with open(os.path.join(args.objdir,args.fname+'.json'),"w") as f:
        json.dump(objects,f)

args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

args.config = 'mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'
args.checkpoint = 'mmdetection/checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth'
model_m2f = init_detector(args.config, args.checkpoint, device=args.device)

datadir = args.dstdir

if os.path.isdir(args.src):
    clip = os.path.basename(args.src)
    args.imgdir = os.path.join(datadir,clip,'images')
    args.maskdir = os.path.join(datadir,clip,'masks')
    args.objdir = os.path.join(datadir,clip,'objects')
    os.makedirs(args.imgdir,exist_ok=True)
    os.makedirs(args.maskdir,exist_ok=True)
    os.makedirs(args.objdir,exist_ok=True)
    
    img_list = []
    for ext in ['*.jpg', '*.png']: 
        img_list.extend(glob(os.path.join(args.src, ext)))
    img_list.sort()
    tempimg = cv2.imread(img_list[0],cv2.IMREAD_COLOR)
    args.h,args.w,_ = tempimg.shape
    for imgpath in tqdm(img_list):
        args.img = imgpath
        args.fname, args.ext = os.path.basename(args.img).split('.')
        segmentation(args, model_m2f)
else:
    print(f"Directory {args.src} not exists.")
    # args.img = args.src
    # args.imgdir = os.path.join(datadir,'single','images')
    # args.maskdir = os.path.join(datadir,'single','masks')
    # os.makedirs(args.imgdir,exist_ok=True)
    # os.makedirs(args.maskdir,exist_ok=True)
    # args.fname, args.ext = os.path.basename(args.img).split('.')
    # tempimg = cv2.imread(args.img,cv2.IMREAD_COLOR)
    # args.h,args.w,_ = tempimg.shape
    # segmentation(args, model_m2f)