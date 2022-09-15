import argparse

parser = argparse.ArgumentParser(description='Instance Segmentation')
# ----------------------------------
parser.add_argument('src', type=str, help='single image or directory')
parser.add_argument('--dstdir', default='dataset', help='mask save dir')
parser.add_argument('--score-thr', type=float, default=0.3, help='bbox score threshold')
parser.add_argument('--area-thr', type=float, default=0.05, help='object area threshold')
# ----------------------------------
args = parser.parse_args()
