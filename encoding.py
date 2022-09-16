import argparse
import cv2
from tqdm import tqdm
from glob import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument('src', help='Frame Directory')
args = parser.parse_args()

viddir = './video'

path = args.src
_,clip,model,mode = args.src.split('/')
dstdir = os.path.join(viddir,clip)
os.makedirs(dstdir,exist_ok=True)
pathOut = os.path.join(dstdir,f'{model}_{mode}.mp4')

img_list = []
for ext in ['*.jpg', '*.png']: 
    img_list.extend(glob(os.path.join(path, ext)))
img_list.sort()

fps = 20
frame_array = []
for idx , path in tqdm(enumerate(img_list)) : 
    img = cv2.imread(path)
    height, width, layers = img.shape
    size = (width,height)
    frame_array.append(img)
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc('m','p','4','v'), fps, size)
# out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(frame_array)):
    out.write(frame_array[i])
out.release()