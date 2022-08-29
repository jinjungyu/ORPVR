import json
import cv2

params={'winshape':cv2.MORPH_CROSS,'ksize':5,'Iter':3}
with open("params.json","w") as f:
    json.dump(params,f)
with open("params.json","r") as f:
    data = json.load(f)
print(data,type(data))