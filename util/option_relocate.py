import argparse

parser = argparse.ArgumentParser(description='Object Relocation')
# ----------------------------------
parser.add_argument('src', type=str, help='single image or directory')
parser.add_argument('--dstdir', default='result', help='result save dir')
parser.add_argument('--mode', choices=[0,1,2,3], type=int, required=True, help='relocation mode')
# ----------------------------------
args = parser.parse_args()

class Relocator:
    def __init__(self,args):
        self.w = args.w
        self.nw = args.new_w
        self.ldot = self.w // 4
        self.rdot = self.w - self.w // 4
        self.offset = self.nw - self.w
        self.offset_half = self.offset // 2
        self.mode = args.mode
    def __call__(self,i,j,bbox):
        if args.mode == 0:
            return i, j
        elif args.mode == 1:
            return i, j + self.offset
        elif args.mode == 2:
            if bbox[0] < 5:
                return i, j
            elif bbox[2] > self.w - 5:
                return i, j + self.offset
            else:
                return i, j + self.offset_half
        elif args.mode == 3: # not implemented
            return i, j