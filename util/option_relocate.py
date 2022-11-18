import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Object Relocation')
# ----------------------------------
parser.add_argument('src', type=str, help='single image or directory')
parser.add_argument('--dstdir', default='result', help='result save dir')
parser.add_argument('--mode', default=2, choices=[0,1,2], type=int, help='relocation mode')
# ----------------------------------
args = parser.parse_args()

class Relocator:
    def __init__(self,args):
        self.w = args.w
        self.nw = args.new_w
        self.offset = self.nw - self.w
        self.offset_half = self.offset // 2
        self.margin = 5
        self.l_line = self.margin
        self.r_line = self.w - self.margin
        self.mode = args.mode
        if self.mode == 2:
            self.pre = None
            self.state_offset = [0, self.offset_half, self.offset]
            self.augment = False
            self.n_augframes = 4
    def relocate(self,bimg,img,objects):
        newimg = bimg.copy()
        if self.mode == 0:
            for coor in objects['coor']:
                for i,j in coor:
                    newimg[i][j] = img[i][j]
        elif self.mode == 1:
            for coor in objects['coor']:
                for i,j in coor:
                    newimg[i][j+self.offset] = img[i][j]
        elif self.mode == 2:
            for k in range(len(objects['box'])):
                bbox,coor = objects['box'][k],objects['coor'][k]
                if bbox[0] < self.l_line:
                    dj = 0
                elif bbox[2] > self.r_line:
                    dj = self.offset
                else:
                    dj = self.offset_half
                for i,j in coor:
                    newimg[i][j+dj] = img[i][j]
        
        return newimg

# deprecated
# from image_similarity_measures.quality_metrics import rmse
# class Relocator:
#     def __init__(self,args):
#         self.w = args.w
#         self.nw = args.new_w
#         self.offset = self.nw - self.w
#         self.offset_half = self.offset // 2
#         self.margin = 5
#         self.l_line = self.margin
#         self.r_line = self.w - self.margin
#         self.mode = args.mode
#         if self.mode == 2:
#             self.pre = None
#             self.state_offset = [0, self.offset_half, self.offset]
#             self.augment = False
#             self.n_augframes = 4
#     def __call__(self,bimg,img,objects):        
#         if self.mode == 0:
#             newimg = bimg.copy()
#             for coor in objects['coor']:
#                 for i,j in coor:
#                     newimg[i][j] = img[i][j]
#             return newimg
#         elif self.mode == 1:
#             newimg = bimg.copy()
#             for coor in objects['coor']:
#                 for i,j in coor:
#                     newimg[i][j+self.offset] = img[i][j]
#             return newimg
#         elif self.mode == 2:
#             num = len(objects['box']) # 현재 프레임 객체 개수
#             # 0 : 왼쪽 1: 중간 2: 오른쪽
#             flags = self.boxes_status(objects['box']) #현재 프레임 객체의 state
#             rois = self.make_rois(img,objects) # 현재 프레임 객체의 유사도 비교 위한 roi 이미지

#             if self.pre is not None:
#                 pre_bimg,pre_img,pre_objects,pre_flags,pre_rois = self.pre
#                 match_ids = self.object_matching(pre_rois,rois)
#                 # print("match_ids",match_ids)
#                 for idx in range(num):
#                     pre_idx = match_ids[idx]
#                     if pre_idx is not None:
#                         if abs(flags[idx] - pre_flags[pre_idx]) == 1: # 가장자리와 중간의 전환이 일어나는 트리거
#                             self.augment = True
#                             break

#             augframes = []
            
#             if self.augment: # 전환 상황이면 여기
#                 list_dx = np.zeros((num,), dtype=int) # 현재 프레임 객체 개수 만큼 속도 칸 생성
#                 for idx in range(num):
#                     # state_offset : 화면 비 변환에 따른 기본 offset
#                     # 이전에 있었던 객체 중 전환이 일어난 것은 속도 추가
#                     pre_idx = match_ids[idx]
#                     if pre_idx is not None:
#                         state = flags[idx]
#                         pre_state = pre_flags[match_ids[idx]]
#                         if abs(state - pre_state) == 1 or state == 1 and pre_state == 1: # 전환 상황이나 둘 다 중간일 때
#                             pre_x1 = pre_objects['box'][pre_idx][0]
#                             pre_state_offset = self.state_offset[pre_flags[pre_idx]]
                            
#                             x1 = objects['box'][idx][0]
#                             state_offset = self.state_offset[flags[idx]]
                            
#                             dx = ((x1 + state_offset) - (pre_x1 + pre_state_offset)) // (self.n_augframes + 1)
#                             list_dx[idx] = dx
#                 # print("post 위치",(x1 + state_offset))
#                 # print("pre 위치",(pre_x1 + pre_state_offset))
#                 # print("pre_boxes",pre_objects['box'])
#                 # print("boxes",objects['box'])
#                 # print("flags",flags)
#                 # print("pre_flags",pre_flags)
#                 # print("list_dx",list_dx)
#                 pre_offsets = np.zeros((len(list_dx),), dtype=int)
#                 post_offsets = list_dx * (-(self.n_augframes // 2 + 1))
#                 for bim,im,obj,offsets,indices,states in [[pre_bimg,pre_img,pre_objects,pre_offsets, match_ids, pre_flags]
#                                                    ,[bimg,img,objects,post_offsets, list(range(num)), flags]]:
#                     for k in range(self.n_augframes // 2):
#                         aug_img = bim.copy()
#                         offsets += list_dx
#                         # pre일 때는 post의 객체 순서로 인덱스가 정렬됐다고 생각.
#                         for l in range(len(indices)):
#                             idx = indices[l]
#                             if idx is not None: # None이면 이전에 없었던 것이거나 가장자리(가정)
#                                 try:
#                                     for i,j in obj['coor'][idx]:
#                                         aug_img[i][j+offsets[idx]+self.state_offset[states[idx]]] = im[i][j]
#                                 except:
#                                     print("flags",flags)
#                                     print("pre_flags",pre_flags)
#                                     print("match_ids",match_ids)
#                                     print("i,j",i,j)
#                                     print("j,offsets[idx],self.state_offset[states[idx]])",j,offsets[idx],self.state_offset[states[idx]])
#                                     print("j+offsets[idx]+self.state_offset[states[idx]]",j+offsets[idx]+self.state_offset[states[idx]])
#                         augframes.append(aug_img)
#                 self.augment = False
#             else:    
#                 newimg = bimg.copy()
#                 for k, coor in enumerate(objects['coor']):
#                     for i,j in coor:
#                         newimg[i][j+self.state_offset[flags[k]]] = img[i][j]
                    
#                 augframes.append(newimg)
            
#             if num == 0:
#                 self.pre = None
#             else:
#                 self.pre = [bimg,img,objects,flags,rois]

#             return augframes
    
#     def make_rois(self,img,objects):
#         boxes = objects['box']
#         coors = objects['coor']

#         num = len(boxes)

#         rois = []

#         # 현재 객체와 과거 객체에 대한 ROI를 모두 추출
#         for i in range(num):
#             x1,y1,x2,y2 = boxes[i]
#             roi = np.zeros_like(img[y1:y2,x1:x2],np.uint8) # roi 크기 만큼 뽑기
#             for r,c in coors[i]:
#                 roi[r-y1][c-x1] = img[r][c]
#             rois.append(roi)
        
#         return rois
    
#     def object_matching(self,pre_rois,rois):

#         pre_num = len(pre_rois)
#         num = len(rois)

#         similarity = [[0] * pre_num for _ in range(num)]

#         # 현재와 과거의 ROI를 비교하여 유사도 메트릭에 저장
#         for i in range(num):
#             h,w = rois[i].shape[:-1]
#             size = h*w
#             for j in range(pre_num):
#                 pre_h,pre_w = pre_rois[j].shape[:-1]
#                 pre_size = pre_h*pre_w
#                 if pre_size < size:
#                     roi = cv2.resize(rois[i],(pre_w,pre_h))
#                     pre_roi = pre_rois[j]
#                 else:
#                     roi = rois[i]
#                     pre_roi = cv2.resize(pre_rois[j],(w,h))
#                 similarity[i][j] = rmse(roi, pre_roi)

#         post_to_pre = np.argmin(np.array(similarity),axis=1)
#         pre_to_post = np.argmin(np.array(similarity),axis=0)
#         match_ids = [None] * num
#         # print("post match_ids", [post_to_pre[i] for i in range(num)])
#         # print("pre match_ids", [pre_to_post[i] for i in range(pre_num)])
#         for i in range(num):
#             match_ids[pre_to_post[post_to_pre[i]]] = post_to_pre[i]
#         # print("match_ids",match_ids)

#         return match_ids

    
#     def boxes_status(self,boxes):
#         edge_flags = [0] * len(boxes)
#         for i in range(len(boxes)):
#             x1,_,x2,_ = boxes[i]
#             if x1 < self.l_line:
#                 flag = 0
#             elif x2 > self.r_line:
#                 flag = 2
#             else:
#                 flag = 1
#             edge_flags[i] = flag
#         return edge_flags
    

if __name__ == '__main__':
    import cv2
    import os
    import json
    from glob import glob
    
    clip = os.path.basename(os.path.dirname(args.src))
    
    bimgdir = args.src
    imgdir = os.path.join('dataset',clip,'images')
    objdir = os.path.join('dataset',clip,'objects')
    
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
    
    n = 15
    for i in range(n,n+3):
        img1 = cv2.cvtColor(cv2.imread(ilist[i],cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(ilist[i+1],cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        mask1 = cv2.imread(ilist[i].replace('images','masks'),cv2.IMREAD_GRAYSCALE)
        mask2 = cv2.imread(ilist[i+1].replace('images','masks'),cv2.IMREAD_GRAYSCALE)
        with open(olist[i],"r") as f:
            objects1 = json.load(f)
        with open(olist[i+1],"r") as f:
            objects2 = json.load(f)
        print(i+1)
        print(objects1['box'],objects2['box'])
        
        bimg = cv2.imread(flist[0],cv2.IMREAD_COLOR)
        args.h,args.w,_ = bimg.shape
        args.new_w = int(np.ceil(args.h * 16 / 9)) # 640 -> 854
        
        relocator = Relocator(args)
        
        # relocator.object_matching(img1,objects1,img2,objects2)
    # plt.figure()
    # plt.subplot(2,2,1)
    # plt.imshow(img1)
    # plt.subplot(2,2,2)
    # plt.imshow(img2)
    # plt.subplot(2,2,3)
    # plt.imshow(mask1,cmap='gray')
    # plt.subplot(2,2,4)
    # plt.imshow(mask2,cmap='gray')
    # plt.tight_layout()
    # plt.show()
    # pre_to_post, post_to_pre = relocator.object_matching(objects1,objects2)
    # print(pre_to_post)
    # print(post_to_pre)