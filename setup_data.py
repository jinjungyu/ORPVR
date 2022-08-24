from argparse import ArgumentParser
from PIL import Image
import os
from shutil import copy
import numpy as np

from mmdetection.mmdet.apis import inference_detector,init_detector

white = np.array([255,255,255])
target = 0 # Person
black = np.array([0,0,0])

def parsing():
    parser = ArgumentParser()
    parser.add_argument('src', help='Image file or Directory')
    parser.add_argument('--sep', action='store_true')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument('--area-thr', type=float, default=0.1, help='bbox score threshold')
    args = parser.parse_args()
    return args

def main(args):
    # 소스는 rawdataset 아래에 있다고 간주
    if args.sep:
        datadir = './dataset2'
    else:
        datadir = './dataset'
    if not os.path.exists(datadir):
        os.mkdir(datadir)

    # Mask2former config
    args.config = 'mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'
    args.checkpoint = 'mmdetection/checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth'

    model = init_detector(args.config, args.checkpoint, device=args.device)

    # 입력이 directory
    # args.src = REDS_640x480/12
    if os.path.isdir(os.path.join('rawdataset',args.src)):
        args.imgdir = os.path.join(datadir,args.src,'images')
        args.maskdir = os.path.join(datadir,args.src,'masks')
        os.makedirs(args.imgdir,exist_ok=True)
        os.makedirs(args.maskdir,exist_ok=True)

        args.src = os.path.join('rawdataset',args.src)
        args.cnt = 1

        dir,_,files = next(iter(os.walk(args.src)))
        for file in sorted(files):
            filepath = os.path.join(dir,file)
            result = inference_detector(model,filepath)
            masking(args,filepath,result)
            args.cnt += 1
    else:
        # args.src = man.jpg or street/man.jpg
        if args.src.endswith(('.jpg','.png')):
            filepath = os.path.join('rawdataset',args.src)
            args.imgdir = os.path.join(datadir,'single','images')
            args.maskdir = os.path.join(datadir,'single','masks')
            os.makedirs(args.imgdir,exist_ok=True)
            os.makedirs(args.maskdir,exist_ok=True)
            args.cnt = None
            result = inference_detector(model, filepath)
            masking(args,filepath,result)
        else:
            print("Input is neither Image file(jpg, png) nor Directory")
            return

def masking_sep(args,filepath,result):
    imgname = os.path.basename(filepath)
    if args.cnt:
        ext = imgname.split('.')[-1]
        fname = f'img{args.cnt:>04}'
    else:
        fname,ext = imgname.split('.')

    copy(filepath,os.path.join(args.imgdir,f'{fname}.{ext}'))

    maskdir = os.path.join(args.maskdir,fname)
    os.makedirs(maskdir,exist_ok=True)
    
    img = Image.open(filepath)
    img = np.array(img)
    h,w = img.shape[:-1]
    n = 1
    for k in range(len(result[0][target])):
        score = result[0][target][k][-1]
        # threshold 보다 낮은 박스는 무시
        if score < args.score_thr:
            continue
        # 박스 내만 돌며 마스킹
        c1,r1,c2,r2 = map(int,result[0][target][k][:-1])
        # print(r1,c1,r2,c2)
        mask = result[1][target][k]
        for i in range(h):
            for j in range(w):
                if mask[i][j]:
                    img[i][j] = white
                else:
                    img[i][j] = black
        img_mask = Image.fromarray(img)
        img_mask.save(os.path.join(maskdir,f'mask{n:>03}.{ext}'))
        n += 1

def masking(args,filepath,result):
    imgname = os.path.basename(filepath)
    if args.cnt:
        ext = imgname.split('.')[-1]
        fname = f'img{args.cnt:>04}'
        maskname = f'mask{args.cnt:>04}'
    else:
        fname,ext = imgname.split('.')
        maskname = fname

    copy(filepath,os.path.join(args.imgdir,f'{fname}.{ext}'))
    
    img = Image.open(filepath)
    img = np.array(img)

    h,w = img.shape[:-1]
    size = h*w
    mask = np.zeros((h,w),dtype=np.uint8)
    coors = set()
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
        # 객체 크기가 threshold 보다 크면 마스킹 포함
        if len(coor) / size > args.area_thr:
            # print((c1,r1),(c2,r2))
            # print(len(coor) / size, args.area_thr)
            coors.update(coor)
    # masking
    for i,j in coors:
        mask[i][j] = 255

    img_mask = Image.fromarray(mask)
    img_mask.save(os.path.join(args.maskdir,f'{maskname}.{ext}'))

if __name__ == "__main__":
    args = parsing()
    main(args)