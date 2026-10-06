"""复用旧项目的zbar解码；只返回完整二维码文本和图像位置。"""
from kick_shapes import Box


def read_codes(frame):
    from pyzbar.pyzbar import decode, ZBarSymbol
    codes = []
    for code in decode(frame, symbols=[ZBarSymbol.QRCODE]):
        text = code.data.decode("utf-8")
        if text:
            codes.append((text, Box(*code.rect)))
    return codes
