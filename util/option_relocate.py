import argparse
from nis import match
import numpy as np

parser = argparse.ArgumentParser(description='Object Relocation')
# ----------------------------------
parser.add_argument('src', type=str, help='single image or directory')
parser.add_argument('--dstdir', default='result', help='result save dir')
parser.add_argument('--mode', choices=[0,1,2,3], type=int, required=True, help='relocation mode')
# ----------------------------------
args = parser.parse_args()

def object_matching(objects1,objects2):
    matching_ids = [None] * len(objects2['box']) # 뒤 프레임 객체 수 만큼.
    for i in range(len(objects2['box'])): # 뒤 프레임 객체를 돌면서 Euclidian Distance
        dists,sizes = [],[]
        post_x1,post_y1,post_x2,post_y2 = objects2['box'][i]
        post_center = np.array(((post_x1 + post_x2)/2,(post_y1 + post_y2)/2))
        post_size = len(objects2['coor'][i])
        for j in range(len(objects1['box'])): # 앞 프레임의 어느 객체에 맞는지 유사도 검사
            pre_x1,pre_y1,pre_x2,pre_y2 = objects1['box'][j]
            pre_center = np.array(((pre_x1 + pre_x2)/2,(pre_y1 + pre_y2)/2))
            pre_size = len(objects1['coor'][j])
            
            dist = np.sum(np.square(post_center-pre_center))
            size_diff = abs(post_size-pre_size)
            dists.append(dist)
            sizes.append(size_diff)
        ids_dists = np.array(sorted(range(len(dists), key=lambda k: dists[k])))
        ids_sizes = np.array(sorted(range(len(sizes), key=lambda k: sizes[k])))
        idx = np.argmin(ids_dists+ids_sizes)
        matching_ids[i] = idx

    return matching_ids

class Relocator:
    def __init__(self,args):
        self.w = args.w
        self.nw = args.new_w
        self.offset = self.nw - self.w
        self.offset_half = self.offset // 2
        self.mode = args.mode
        if self.mode == 2:
            self.pre = None
            self.edge = False
    def __call__(self,i,j,bbox):
        if self.mode == 0:
            return i, j
        elif self.mode == 1:
            return i, j + self.offset
        elif self.mode == 2:
            if bbox[0] < 5:
                self.edge = True
                return i, j
            elif bbox[2] > self.w - 5:
                self.edge = True
                return i, j + self.offset
            else:
                self.edge = False
                return i, j + self.offset_half
            self.pre = 
        elif self.mode == 3: # not implemented
            return i, j