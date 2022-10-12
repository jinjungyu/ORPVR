import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import animation
import argparse
from glob import glob

models = ['aotgan','e2fgvi_hq']
modes= ['original','offset','dynamic']

parser = argparse.ArgumentParser()
parser.add_argument('clip', help='clipname')
args = parser.parse_args()

videos=[]
srcdir = './result'
for model in models:
    for mode in modes:
        clipdir = os.path.join(srcdir,args.clip,model,mode)
        for ext in ['*.png','*.jpg']:
            fnames = glob(os.path.join(clipdir,ext))
        fnames.sort()
        frames = [Image.open(f) for f in fnames]
        videos.append(frames)

fig = plt.figure(figsize=(12,8))
imdatas=[]
for i in range(6):
    ax = fig.add_subplot(2, 3, i+1)
    ax.axis('off')
    ax.set_title(f'{models[i//3]}_{modes[i%3]}')
    imdatas.append(ax.imshow(videos[i][0]))

def update(idx):
    for i in range(6):
        imdatas[i].set_data(videos[i][idx])

fig.tight_layout()
anim = animation.FuncAnimation(fig,
                                update,
                                frames=len(frames),
                                interval=50)
plt.show()