"""蓝色连到地面时，从明暗边界补充盒面候选。"""
import cv2
import numpy as np
from kick_shapes import Box


def supported_blue_boundary(frame, box):
    """完整候选至少三条边有独立图像边缘，不只依赖颜色阈值边界。"""
    edges=cv2.Canny(cv2.GaussianBlur(frame,(5,5),0),40,100)
    x,y,w,h=map(int,(box.x,box.y,box.width,box.height))
    edges=cv2.dilate(edges,np.ones((7,7),np.uint8))
    sides=(edges[y,x:x+w],edges[y+h-1,x:x+w],
           edges[y:y+h,x],edges[y:y+h,x+w-1])
    return sum(np.mean(side>0)>=.35 for side in sides)>=3


def edge_blue_boxes(frame):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lab=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)
    # 保留亮度边界，不用“所有蓝色像素连通”决定盒面范围。
    edges=cv2.Canny(cv2.GaussianBlur(frame,(5,5),0),40,100)
    edges=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2]
    height,width=frame.shape[:2]
    boxes=[]
    for contour in contours:
        polygon=cv2.approxPolyDP(contour,.025*cv2.arcLength(contour,True),True)
        if len(polygon)!=4 or not cv2.isContourConvex(polygon):
            continue
        x,y,w,h=cv2.boundingRect(polygon)
        if min(w,h)<20 or not .5<=w/h<=2 or w*h>width*height*.45:
            continue
        if x<3 or y<3 or x+w>=width-3 or y+h>=height-3:
            continue
        inset=max(2,min(w,h)//8)
        inner=hsv[y+inset:y+h-inset,x+inset:x+w-inset]
        blue=(inner[:,:,0]>=85)&(inner[:,:,0]<=130)&(inner[:,:,1]>=80)&(inner[:,:,2]>=60)
        if blue.mean()<.65:
            continue
        margin=max(4,min(w,h)//8)
        left,top=max(0,x-margin),max(0,y-margin)
        right,bottom=min(width,x+w+margin),min(height,y+h+margin)
        ring=np.ones((bottom-top,right-left),bool)
        ring[y-top:y+h-top,x-left:x+w-left]=False
        inside=np.median(lab[y+inset:y+h-inset,x+inset:x+w-inset],axis=(0,1))
        outside=np.median(lab[top:bottom,left:right][ring],axis=0)
        if np.linalg.norm(inside-outside)<18:
            continue
        boxes.append(Box(x,y,w,h))
    return boxes
