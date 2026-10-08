"""保留二维码四角；与实测投放视角比较，不从斜视猜测横移方向。"""
from dataclasses import dataclass
import cv2
import numpy as np
from kick_shapes import Box


@dataclass(frozen=True)
class QRBox(Box):
    corners: tuple

    def normalized(self, shape):
        height, width = shape[:2]
        return QRBox(self.x/width,self.y/height,self.width/width,self.height/height,
                     tuple((x/width,y/height) for x,y in self.corners))


def qr_box(corners):
    points=np.asarray(corners,np.float32).reshape(-1,2)
    hull=cv2.convexHull(points)
    polygon=cv2.approxPolyDP(hull,.02*cv2.arcLength(hull,True),True).reshape(-1,2)
    x,y,w,h=cv2.boundingRect(points)
    if len(polygon)!=4:
        return Box(x,y,w,h)
    center=polygon.mean(axis=0)
    order=np.argsort(np.arctan2(polygon[:,1]-center[1],polygon[:,0]-center[0]))
    polygon=polygon[order]
    polygon=np.roll(polygon,-int(np.argmin(polygon.sum(axis=1))),axis=0)
    return QRBox(x,y,w,h,tuple((float(px),float(py)) for px,py in polygon))


def quad_signature(corners, shape):
    # 还原像素长宽比例；归一化坐标中直接量边会受640x480比例影响。
    points=np.asarray(corners,float)*np.array([shape[1],shape[0]])
    lengths=np.linalg.norm(points-np.roll(points,-1,axis=0),axis=1)
    return lengths/lengths.mean()


def view_matches(box, reference, camera):
    key='drop_quad' if camera=='belly' else 'drop_head_quad'
    target=reference.get(key)
    if target is None:
        return True  # 旧标定仍使用原宽高检查，不伪造四角。
    corners=getattr(box,'corners',None)
    if corners is None:
        return False
    shape=reference['shapes'][camera]
    current=quad_signature(corners,shape)
    expected=quad_signature(target,shape)
    # 比较相对边长，允许整体缩放，拒绝明显不同的透视形状。
    return bool(np.all(np.abs(current-expected)<=.18))


def quad_size_ratios(box, reference, camera):
    key='drop_quad' if camera=='belly' else 'drop_head_quad'
    if key not in reference or not hasattr(box,'corners'):
        return None
    shape=reference['shapes'][camera]
    def sides(corners):
        points=np.asarray(corners,float)*np.array([shape[1],shape[0]])
        lengths=np.linalg.norm(points-np.roll(points,-1,axis=0),axis=1)
        return np.array([(lengths[0]+lengths[2])/2,(lengths[1]+lengths[3])/2])
    return sides(box.corners)/sides(reference[key])
