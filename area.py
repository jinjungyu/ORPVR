import cv2
import numpy as np
import os
from argparse import ArgumentParser
from PIL import Image

from mmdetection.mmdet.apis import inference_detector,init_detector

def parsing():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args

params = {}

def main(args):
    
    args.config = 'mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'
    args.checkpoint = 'mmdetection/checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth'
     
    model = init_detector(args.config, args.checkpoint, device=args.device)
    result = inference_detector(model,args.img)

    target = 0 # Person
    
    img = Image.open(args.img)
    img = np.array(img)

    h,w = img.shape[:-1]
    size = h*w
    mask = np.zeros((h,w),dtype=np.float32)
    # coors = set()
    
    for k in range(len(result[0][target])):
        score = result[0][target][k][-1]
        # score threshold 보다 낮은 박스는 무시
        if score < args.score_thr:
            continue
        coor = set()
        c1,r1,c2,r2 = map(int,result[0][target][k][:-1])
        temp = result[1][target][k]
        for i in range(r1-1,r2):
            for j in range(c1-1,c2):
                if temp[i][j]:
                    coor.add((i,j))
        
        rate = (len(coor) / size) * 100
        for i,j in coor:
            mask[i][j] = rate
    
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    
    def area_cutting(img:np.ndarray,mask:np.ndarray):
        global params
        cv2.namedWindow('Image')
        cv2.namedWindow('Mask')
        
        cv2.imshow('Image',img)
        cv2.createTrackbar('Area_Threshold','Mask',0,10000,lambda x:x)
        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            
            a = cv2.getTrackbarPos('Area_Threshold','Mask')
            a /= 100 # 0.00 ~ 100.00
            
            img_mask = cv2.threshold(mask,a,255,cv2.THRESH_BINARY)[1]
            cv2.imshow('Mask',img_mask)
        cv2.destroyAllWindows()
        params['area_thr']=a
        return img_mask
    mask = area_cutting(img,mask)
    
    def addmargin(mask):
        global params
        cv2.namedWindow('Mask_origin')
        cv2.namedWindow('Mask_margin')
        
        cv2.imshow('Mask_origin',mask)
        cv2.createTrackbar('Iter','Mask_margin',0,10,lambda x:x)
        cv2.createTrackbar('Size','Mask_margin',6,14,lambda x:x)
        cv2.setTrackbarMin('Size','Mask_margin',6)
        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            if k == ord('s'):
                dstdir = 'test'
                if not os.path.exists(dstdir):
                    os.mkdir(dstdir)
                ext = args.img.split('.')[-1]
                fname = input("Save File name : ")
                filepath = os.path.join(dstdir,f'{fname}.{ext}')
                cv2.imwrite(filepath,mask_margin)
                print("result is saved in",filepath)
                break
            k_size = cv2.getTrackbarPos('Size','Mask_margin') // 2
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size,k_size))
            n = cv2.getTrackbarPos('Iter','Mask_margin')
            mask_margin = cv2.dilate(mask, kernel, iterations = n)
            cv2.imshow('Mask_margin',mask_margin)
        cv2.destroyAllWindows()
        params['ksize'] = k_size
        params['Iter'] = n
        return mask_margin
    mask_margin = addmargin(mask)
    
    def comparing(img,mask,mask_margin):
        global params
        
        mask = mask > 0
        mask_margin = mask_margin > 0
        img1 = img.copy()
        img2 = img.copy()
        
        for i in range(h):
            for j in range(w):
                if mask[i][j]:
                    img1[i][j] = (255,255,255)
                    
        for i in range(h):
            for j in range(w):
                if mask_margin[i][j]:
                    img2[i][j] = (255,255,255)
                    
        img_merged = np.concatenate([img1,img2],axis=1)
        cv2.imshow('Compare',img_merged)
        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()
    comparing(img,mask,mask_margin)

    print("Result")
    for k,v in params.items():
        print(f"{k} : {v}")
    print("Done")
if __name__ == '__main__':
    args = parsing()
    main(args)