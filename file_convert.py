from glob import glob
import os
import glob

files = glob.glob("vid_inp2/inpaint_res/*.png")
for name in files:
    if not os.path.isdir(name):
        src = os.path.splitext(name)
        os.rename(name,src[0]+'.jpg')