import argparse

parser = argparse.ArgumentParser(description='Image Inpainting')
# ----------------------------------
parser.add_argument('src', type=str, help='single image or directory')
parser.add_argument('--model', default='e2fgvi_hq', choices=['aotgan','e2fgvi_hq'], help='inpainting model')
parser.add_argument('--dstdir', default='result_inpaint', help='result save dir')
parser.add_argument('--device', type=str, default=None, help='cpu or cuda')
# ----------------------------------
args = parser.parse_args()