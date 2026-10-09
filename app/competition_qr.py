"""复用zbar解码；只返回完整二维码文本和原始图像位置。"""
import cv2
from kick_shapes import Box
from qr_geometry import qr_box, QRBox
import numpy as np


def partial_edge_qr(frame):
    """找边缘部分码：定位方框加附近模块纹理；只判外观，不推断文本。"""
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,31,5)
    contours,hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None
    height,width = frame.shape[:2]
    finders = []
    for index,contour in enumerate(contours):
        child = hierarchy[0,index,2]
        grandchild = hierarchy[0,child,2] if child >= 0 else -1
        if child < 0:
            continue
        area = cv2.contourArea(contour)
        if area < 64:
            continue
        polygon = cv2.approxPolyDP(contour,.04*cv2.arcLength(contour,True),True)
        x,y,w,h = cv2.boundingRect(contour)
        if min(w,h) < 8 or not .5 <= w/h <= 2:
            continue
        # 定位图案由外黑框、内白框、中心黑块嵌套组成。
        intact = (len(polygon) == 4 and cv2.isContourConvex(polygon)
                  and grandchild >= 0
                  and .25 <= cv2.contourArea(contours[child])/area <= .9
                  and .08 <= cv2.contourArea(contours[grandchild])/area <= .55)
        # 出画或模糊可破坏中心黑块嵌套；边缘残框仍须有内白孔和邻近模块。
        inner = contours[child]
        inner_polygon = cv2.approxPolyDP(inner,.04*cv2.arcLength(inner,True),True)
        edge = min(x,y,width-x-w,height-y-h) <= 3
        partial = (edge and 4 <= len(polygon) <= 6
                   and 4 <= len(inner_polygon) <= 6
                   and .15 <= cv2.contourArea(inner)/area <= .8
                   and area/cv2.contourArea(cv2.convexHull(contour)) >= .6)
        if not intact and not partial:
            continue
        finders.append(Box(x,y,w,h))
    candidates = []
    for finder in finders:
        size = (finder.width+finder.height)/2
        if min(finder.x,finder.y,width-finder.x-finder.width,height-finder.bottom) > 2*size:
            continue
        modules = 0
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            cx,cy = x+w/2,y+h/2
            if finder.x <= cx <= finder.x+finder.width and finder.y <= cy <= finder.bottom:
                continue
            if abs(cx-finder.cx) > 3*size or abs(cy-finder.cy) > 2*size:
                continue
            if (.08*size <= min(w,h) and max(w,h) <= .65*size
                    and .4 <= w/h <= 2.5 and cv2.contourArea(contour) >= .007*size*size):
                modules += 1
        # 只剩一个完整定位框时，还要有足够的邻近黑白模块，孤立方框不触发。
        if len(finders) == 1 and modules >= 6:
            candidates.append(finder)
    for index,first in enumerate(finders):
        for second in finders[index+1:]:
            size = (first.width+first.height+second.width+second.height)/4
            distance = np.hypot(first.cx-second.cx,first.cy-second.cy)
            if not .5 <= first.width*first.height/(second.width*second.height) <= 2:
                continue
            if not 1.5*size <= distance <= 8*size:
                continue
            left,top = min(first.x,second.x),min(first.y,second.y)
            right,bottom = max(first.x+first.width,second.x+second.width),max(first.bottom,second.bottom)
            # 被截断的码，仍可见的定位框应靠近画面边缘。
            margin = 2*size
            if min(left,top,width-right,height-bottom) > margin:
                continue
            candidates.append(Box(left,top,right-left,bottom-top))
    # 三个定位框等情况可形成多个框，合并相交的候选；两个不同码则不触发。
    if not candidates:
        return None
    candidate = candidates[0]
    for other in candidates[1:]:
        if (other.x > candidate.x+candidate.width or candidate.x > other.x+other.width
                or other.y > candidate.bottom or candidate.y > other.bottom):
            return None
        left,top = min(candidate.x,other.x),min(candidate.y,other.y)
        right,bottom = max(candidate.x+candidate.width,other.x+other.width),max(candidate.bottom,other.bottom)
        candidate = Box(left,top,right-left,bottom-top)
    return candidate


def qr_images(frame):
    """原图失败后再增强，避免每帧都做全部处理。"""
    yield frame,1
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    enlarged = cv2.resize(gray,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    yield enlarged,2
    binary = cv2.adaptiveThreshold(enlarged,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY,31,5)
    yield binary,2
    # 光照不均时补一次局部对比度增强；仍只接受真正解码出的文本。
    enhanced = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray)
    yield enhanced,1


def read_codes(frame):
    from pyzbar.pyzbar import decode, ZBarSymbol
    for image,scale in qr_images(frame):
        codes = []
        for code in decode(image,symbols=[ZBarSymbol.QRCODE]):
            text = code.data.decode("utf-8")
            if text:
                # 放大图上的框必须还原，否则会错误判断方向和放下位置。
                x,y,w,h = code.rect
                polygon = [(point.x/scale,point.y/scale) for point in code.polygon]
                box = Box(x/scale,y/scale,w/scale,h/scale)
                if len(polygon)>=4:
                    geometry = qr_box(polygon)
                    if isinstance(geometry,QRBox):
                        box = QRBox(box.x,box.y,box.width,box.height,geometry.corners)
                codes.append((text,box))
        if codes: return codes
    return []
