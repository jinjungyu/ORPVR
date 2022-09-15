import argparse

parser = argparse.ArgumentParser(description='Image Inpainting')
# ----------------------------------
parser.add_argument('src', type=str, help='single image or directory')
parser.add_argument('--dstdir', default='result_inpaint', help='result save dir')
parser.add_argument('--model', choices=['aotgan','e2fgvi','e2fgvi_hq'], help='inpainting model')
# ----------------------------------
args = parser.parse_args()