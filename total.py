import cv2
import os
import importlib
import numpy as np
# from argparse import ArgumentParser
from glob import glob
from PIL import Image

import torch
from torchvision.transforms import ToTensor

from mmdetection.mmdet.apis import inference_detector,init_detector

target = 0 # person

def segmentation(args,model):
    result = inference_detector(model,args.img)
    img = cv2.imread(args.img,cv2.IMREAD_COLOR)

    h,w,_ = img.shape
    size = h*w
    # mask = np.zeros((h,w),dtype=np.float32)
    mask = np.zeros((h,w),dtype=np.uint8)
    
    objects = {'box':[],'mask':[]}
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
        for i,j in coor:
            mask[i][j] = 255
        objects['box'].append([c1,r1,c2,r2])
        objects['mask'].append(temp)
        # for area_cutting
        # rate = (len(coor) / size) * 100
        # for i,j in coor:
        #     mask[i][j] = rate
    return img,mask,objects
        
def postprocess(image):
    image = torch.clamp(image, -1., 1.)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return image

def inpainting(args,img,mask,model):
    # load images
    img_tensor = (ToTensor()(img) * 2.0 - 1.0).unsqueeze(0)
    h, w, c = img.shape
    mask = np.reshape(mask, (h,w,1))
    print('[**] inpainting ... ')

    with torch.no_grad():
        mask_tensor = (ToTensor()(mask)).unsqueeze(0)
        masked_tensor = (img_tensor * (1 - mask_tensor).float()) + mask_tensor
        # print(f"mask info: {masked_tensor.shape}")
        # print(f"image info: {img_tensor.shape}")

        pred_tensor = model(masked_tensor, mask_tensor)
        # print(f"pred tensor: {pred_tensor.shape}")
        comp_tensor = (pred_tensor * mask_tensor + img_tensor * (1 - mask_tensor))
        comp_np = postprocess(comp_tensor[0])
    
    return comp_np

def retargeting(img:np.ndarray):
    h,w,_ = img.shape
    new_w = (h * 16) // 9
    return cv2.resize(img,dsize=(new_w,h),interpolation=cv2.INTER_CUBIC)

def repainting(args,img_org,img_inpainted,objects):
    result_dir = 'result'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    filename = os.path.basename(args.img).split('.')[0]
    w = img_org.shape[1]
    new_w = img_inpainted.shape[1]
    diff = (new_w - w) // 2
    for k in range(len(objects['box'])):
        c1,r1,c2,r2 = objects['box'][k]
        mk = objects['mask'][k]
        for i in range(r1-1,r2):
            for j in range(c1-1,c2):
                if mk[i][j]:
                    img_inpainted[i][j+diff] = img_org[i][j]
    cv2.namedWindow('Result')
    img_merged = np.concatenate([img_org,img_inpainted],axis=1)
    cv2.imshow('Result',img_merged)
    while True:
        k = cv2.waitKey(100) & 0xFF
        if k == 27:
            cv2.imwrite(os.path.join(result_dir,filename+"_inpainted.png"), img_inpainted)
            print('inpainting finish!')
            print('[**] save successfully!')
            break

def area_cutting(img:np.ndarray,mask:np.ndarray):
    global params
    cv2.namedWindow('Image')
    cv2.namedWindow('Mask')
    
    cv2.imshow('Image',img)
    cv2.createTrackbar('Area_Threshold','Mask',0,10000,lambda x:x)
    while True:
        k = cv2.waitKey(50) & 0xFF
        if k == 27:
            break
        
        a = cv2.getTrackbarPos('Area_Threshold','Mask')
        a /= 100 # 0.00 ~ 100.00
        
        img_mask = cv2.threshold(mask,a,255,cv2.THRESH_BINARY)[1]
        cv2.imshow('Mask',img_mask)
    cv2.destroyAllWindows()
    params['area_thr']=a
    return img_mask

def addmargin(mask):
    global params
    winshape = [cv2.MORPH_CROSS,cv2.MORPH_RECT,cv2.MORPH_ELLIPSE]
    win='0:Cross\n1:Ellipse\n2:Rect'
    cv2.namedWindow('Mask_origin')
    cv2.namedWindow('Mask_margin')
    
    cv2.imshow('Mask_origin',mask)
    cv2.createTrackbar('Iter','Mask_margin',0,10,lambda x:x)
    cv2.createTrackbar(win,'Mask_margin',0,2,lambda x:x)
    cv2.createTrackbar('Size','Mask_margin',3,7,lambda x:x)
    cv2.setTrackbarMin('Size','Mask_margin',3)
    while True:
        k = cv2.waitKey(50) & 0xFF
        if k == 27:
            break
        k_size = cv2.getTrackbarPos('Size','Mask_margin')
        winidx = cv2.getTrackbarPos(win,'Mask_margin')
        n = cv2.getTrackbarPos('Iter','Mask_margin')
        kernel = cv2.getStructuringElement(winshape[winidx], (k_size,k_size))
        mask_margin = cv2.dilate(mask, kernel, iterations = n)
        cv2.imshow('Mask_margin',mask_margin)
    cv2.destroyAllWindows()
    params['winshape'] = winshape[winidx]
    params['ksize'] = k_size
    params['Iter'] = n
    return mask_margin

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
        k = cv2.waitKey(50) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()

params = {}

def main(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Mask2Former Model
    args.config = 'mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'
    args.checkpoint = 'mmdetection/checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth'
     
    model_m2f = init_detector(args.config, args.checkpoint, device=args.device)
    
    # AOT-GAN Model
    net_aot = importlib.import_module('AOT-GAN-for-Inpainting.src.model.'+args.model)
    model_aot = net_aot.InpaintGenerator(args)
    model_aot.load_state_dict(torch.load("AOT-GAN-for-Inpainting/experiments/G0000000.pt", map_location=args.device))
    model_aot.eval()
    
    # img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    img_org,mask,objects = segmentation(args, model_m2f)
    mask_margin = addmargin(mask)
    img_inpainted = inpainting(args,img_org,mask_margin,model_aot)
    img_retarget = retargeting(img_inpainted)
    img_result = repainting(args,img_org,img_retarget,objects)
    
    # print("Result")
    # for k,v in params.items():
    #     print(f"{k} : {v}")
    # print("Done")
if __name__ == '__main__':
    opt = importlib.import_module('AOT-GAN-for-Inpainting.src.utils.option')
    args = opt.args
    main(args)