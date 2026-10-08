"""复用zbar解码；只返回完整二维码文本和原始图像位置。"""
import cv2
from kick_shapes import Box
from qr_geometry import qr_box, QRBox


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
