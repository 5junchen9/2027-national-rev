"""比赛人脸取景与姓名局部OCR；不改变同学原文件。"""
import re
import cv2


def read_local_name(ocr, image, face_box):
    x, y, width, height = map(int, face_box)
    ih, iw = image.shape[:2]
    left, right = max(0, x-width//2), min(iw, x+width*3//2)
    top, bottom = max(0, y+height), min(ih, y+height*5//2)
    if right <= left or bottom <= top:
        return None, None
    region = image[top:bottom, left:right]
    region = cv2.resize(region, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    candidates = []
    for text, score in ocr._detect_pairs(region):
        name = re.sub(r'^姓名[：:]?', '', str(text).strip())
        name = re.sub(r'\s+', '', name)
        if re.fullmatch(r'[\u4e00-\u9fff]{2,4}', name) and score >= .8:
            candidates.append((name, score))
    return max(candidates, key=lambda item: item[1]) if candidates else (None, None)


def tracked_head_position(position, start_position, face, image_shape):
    if face is None:
        return position
    _, y, _, height = face[:4]
    center = (y + height/2) / image_shape[0]
    # 给脸下方的姓名留空间；每次只调1个单位，围绕实测头位限制±6。
    if center > .55:
        return min(180, start_position+6, position+1)
    if center < .28:
        return max(85, start_position-6, position-1)
    return position
