import cv2
import os
from glob import glob

srcdir = 'rawdataset/DAVIS-2017_semantics-480p/DAVIS/Annotations_semantics/480p/'
dstdir = 'rawdataset/MASK_DAVIS_640x480/'
os.makedirs(dstdir,exist_ok=True)
w = 854
neww = 640
centerw = w // 2
offsetw = neww // 2
exts = ['*.png','*.jpg']
for clip in os.listdir(srcdir):
    clipdst = os.path.join(dstdir,clip)
    os.makedirs(clipdst,exist_ok=True)
    clipsrc = os.path.join(srcdir,clip)
    imlist = os.listdir(clipsrc)
    for imname in imlist:
        img = cv2.imread(os.path.join(clipsrc,imname))
        newimg = img[:,centerw-offsetw:centerw+offsetw]
        cv2.imwrite(os.path.join(clipdst,imname), newimg)
    print(f'{clipsrc} complete')