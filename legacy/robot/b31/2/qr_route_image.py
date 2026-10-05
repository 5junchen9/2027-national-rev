"""提取白底黑线、水平/竖直、无交叉路线。仅使用已有 OpenCV 和标准库。"""
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
import math

import cv2
import numpy as np


class Images(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []

    def handle_starttag(self, tag, attrs):
        if tag == "img" and dict(attrs).get("src"):
            self.sources.append(dict(attrs)["src"])


def fetch_image(url):
    # 网页只解析 img 标签，不执行脚本；只接收 HTTP(S) 图片。
    for attempt in range(2):
        if urlparse(url).scheme not in ("http", "https"):
            raise ValueError("图片地址必须是 HTTP(S)")
        request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html,image/*;q=0.9,*/*;q=0.8"})
        with urlopen(request, timeout=15) as response:
            data = response.read(10 * 1024 * 1024 + 1)
            final_url = response.geturl()
        if len(data) > 10 * 1024 * 1024:
            raise ValueError("下载内容超过 10 MB")
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            return image
        if attempt == 0:
            parser = Images()
            parser.feed(data.decode("utf-8", errors="replace"))
            if len(parser.sources) != 1:
                raise ValueError("网页必须包含一张明确的路线图片")
            url = urljoin(final_url, parser.sources[0])
    raise ValueError("下载内容不是支持的路线图片")


def extract_route(image, longest_steps=8, reverse=False):
    if not 1 <= longest_steps <= 100:
        raise ValueError("最长直线动作次数必须为 1 至 100")
    scale = min(1.0, 1000 / max(image.shape[:2]))
    image = cv2.resize(image, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        raise ValueError("未找到黑色路线")
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.uint8(labels == largest) * 255
    kernel_length = max(15, min(mask.shape) // 20)
    segments = []
    widths = []
    for horizontal in (True, False):
        size = (kernel_length, 1) if horizontal else (1, kernel_length)
        lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, size))
        n, _, boxes, _ = cv2.connectedComponentsWithStats(lines)
        for x, y, w, h, area in boxes[1:]:
            length, thickness = (w, h) if horizontal else (h, w)
            if length < kernel_length or length < thickness * 4:
                continue
            if horizontal:
                segments.append(((float(x), y + (h - 1) / 2), (float(x + w - 1), y + (h - 1) / 2)))
            else:
                segments.append(((x + (w - 1) / 2, float(y)), (x + (w - 1) / 2, float(y + h - 1))))
            widths.append(thickness)
    if not 2 <= len(segments) <= 30:
        raise ValueError("路线必须由 2 至 30 段水平/竖直线组成")
    tolerance = max(6, float(np.median(widths)) * 2)
    # 内部交叉不属于当前支持的单一路线。
    for a, b in segments:
        if abs(a[1] - b[1]) > 1:
            continue
        for c, d in segments:
            if abs(c[0] - d[0]) < 1 and a[0] + tolerance < c[0] < b[0] - tolerance and c[1] + tolerance < a[1] < d[1] - tolerance:
                raise ValueError("路线有交叉，请使用无分支路线图")
    nodes, groups, edges = [], [], []
    for segment in segments:
        ids = []
        for point in segment:
            node = next((i for i, p in enumerate(nodes) if math.dist(point, p) <= tolerance), None)
            if node is None:
                node = len(nodes)
                nodes.append(point)
                groups.append([point])
            else:
                groups[node].append(point)
            ids.append(node)
        edges.append(ids)
    nodes = [tuple(np.mean(points, axis=0)) for points in groups]
    graph = [[] for _ in nodes]
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)
    ends = [i for i, neighbors in enumerate(graph) if len(neighbors) == 1]
    if len(ends) != 2 or any(len(neighbors) not in (1, 2) for neighbors in graph):
        raise ValueError("无法确定单一路线：可能有断线、分支或倾斜")
    current = min(ends, key=lambda i: nodes[i][0])
    if reverse:
        current = next(i for i in ends if i != current)
    order, previous = [], None
    while current is not None:
        if current in order:
            raise ValueError("路线形成闭环")
        order.append(current)
        following = [i for i in graph[current] if i != previous]
        previous, current = current, following[0] if following else None
    if len(order) != len(nodes):
        raise ValueError("路线存在未连接的线段")
    points = [nodes[i] for i in order]
    vectors = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])]
    lengths = [math.hypot(*v) for v in vectors]
    tokens = []
    for i, length in enumerate(lengths):
        steps = round(length / max(lengths) * longest_steps)
        if steps < 1:
            raise ValueError("短线不足一次前进动作，请增加 --longest-steps")
        tokens.append("F" + str(steps))
        if i + 1 < len(vectors):
            a, b = vectors[i:i + 2]
            cosine = (a[0] * b[0] + a[1] * b[1]) / (lengths[i] * lengths[i + 1])
            if abs(cosine) > 0.15:
                raise ValueError("检测到非直角拐弯")
            tokens.append("R90" if a[0] * b[1] - a[1] * b[0] > 0 else "L90")
    for i, point in enumerate(points):
        p = tuple(round(v) for v in point)
        cv2.circle(image, p, 7, (0, 0, 255), -1)
        cv2.putText(image, "START" if i == 0 else str(i), (p[0] + 8, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if i:
            cv2.arrowedLine(image, tuple(round(v) for v in points[i - 1]), p, (0, 0, 255), 2)
    return ",".join(tokens), image
