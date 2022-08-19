import cv2
import numpy as np

# cv2.createTrackbar(trackbarname,winname,value,count,onChange)
# value: 트랙바 시작 값
# count: 트랙바 끝 값
# onChange: 트랙바 이벤트 발생시 수행되는 TrackbakCallback
# TrackbarCallback::def onChange(x)
# cv2.getTrackbarPos(trackbarname,winname) -> retval
# 트랙바의 현재 위치 반환하는 함수
drawing=False
ix,iy=-1,-1
b,g,r=0,0,0

def onMouse(event,x,y,flags,img):
    global drawing,ix,iy,b,g,r
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing=True
        ix,iy=x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.rectangle(img,(ix,iy),(x,y),(b,g,r),-1)
            cv2.imshow('Color_Palette',img)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing=False
def onChange(x):
    pass
def trackbar():
    global b,g,r
    img = np.full((200,512,3),255,np.uint8)
    cv2.namedWindow('Color_Palette')

    cv2.createTrackbar('B','Color_Palette',0,255,onChange)
    cv2.createTrackbar('G','Color_Palette',0,255,onChange)
    cv2.createTrackbar('R','Color_Palette',0,255,onChange)
    switch ='0:OFF\n1:ON'
    cv2.createTrackbar(switch,'Color_Palette',0,1,onChange)
    cv2.setMouseCallback('Color_Palette',onMouse,img)
    while True:
        cv2.imshow('Color_Palette',img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        b=cv2.getTrackbarPos('B','Color_Palette')
        g=cv2.getTrackbarPos('G','Color_Palette')
        r=cv2.getTrackbarPos('R','Color_Palette')
        s=cv2.getTrackbarPos(switch,'Color_Palette')
        if s == 1:
            img[:] = 255
    cv2.destroyAllWindows()
trackbar()