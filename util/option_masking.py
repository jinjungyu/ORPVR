import argparse

parser = argparse.ArgumentParser(description='Instance Segmentation')
# ----------------------------------
parser.add_argument('src', type=str, help='single image or directory')
parser.add_argument('--dstdir', default='dataset', help='mask save dir')
parser.add_argument('--score-thr', type=float, default=0.3, help='bbox score threshold')
parser.add_argument('--area-thr', type=float, default=0.01, help='object area threshold')
parser.add_argument('--device', type=str, default=None, help='cpu or cuda')
# ----------------------------------
args = parser.parse_args()

def compute_intersect_area(box, subbox):
    
    x1, y1, x2, y2 = box 
    x3, y3, x4, y4 = subbox

    ## case1 오른쪽으로 벗어나 있는 경우

    if x2 < x3:
        return 0

    ## case2 왼쪽으로 벗어나 있는 경우
    if x1 > x4:
        return 0

    ## case3 위쪽으로 벗어나 있는 경우
    if  y2 < y3:
        return 0

    ## case4 아래쪽으로 벗어나 있는 경우
    if  y1 > y4:
        return 0

    left_up_x = max(x1, x3)
    left_up_y = max(y1, y3)
    right_down_x = min(x2, x4)
    right_down_y = min(y2, y4)

    width = right_down_x - left_up_x
    height =  right_down_y - left_up_y
  
    return width * height