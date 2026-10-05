import datetime
import json
import os
import time
import numpy as np
import cv2
from typing import Dict, Optional, Tuple

# ---------------------------- 配置文件管理 ----------------------------
class ConfigManager:
    @staticmethod
    def get_config_path() -> str:
        """根据操作系统返回配置文件路径"""
        if os.sep == "/":
            path = "/home/pi/Desktop/b3/Line/img/"
        else:
            path = "F:\\_AAA-BXL\\IMG\\"

    @staticmethod
    def load_config(color_name: str) -> Optional[Dict]:
        """加载颜色配置"""
        path = os.path.join(ConfigManager.get_config_path(), f"{color_name}hsv.json")
        try:
            with open(path) as f:
                loaded_data = json.load(f)
                return {
                    "lower": np.array(loaded_data["lower"]),
                    "upper": np.array(loaded_data["upper"]),
                    "count": loaded_data.get("count", 1),
                    "last_updated": loaded_data.get("last_updated")
                }
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"加载 {color_name} 配置失败: {e}")
            return None

    @staticmethod
    def save_config(color_name: str, threshold: Dict) -> bool:
        """保存颜色配置"""
        os.makedirs(ConfigManager.get_config_path(), exist_ok=True)
        path = os.path.join(ConfigManager.get_config_path(), f"{color_name}hsv.json")
        
        config_data = {
            "lower": threshold["lower"].tolist(),
            "upper": threshold["upper"].tolist(),
            "count": threshold.get("count", 1),
            "last_updated": threshold.get("last_updated", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        }
        
        try:
            with open(path, 'w') as f:
                json.dump(config_data, f, indent=4)
            return True
        except IOError as e:
            print(f"保存 {color_name} 配置失败: {e}")
            return False

# ---------------------------- 颜色检测核心类 ----------------------------
class ColorDetector:
    def __init__(self):
        self.color_thresholds: Dict[str, Dict] = {}  # 存储多颜色阈值
        self.sample_size = 20  # 默认采样区域大小
        self.current_color = "black"  # 当前操作的颜色
        self.last_roi = None  # 最后采样的区域坐标
        self._hsv_cache = None
        self._warned_colors = set()  # 新增：记录已警告的颜色
        self.sample_history = {}  # 新增：存储历史采样数据
        self.sample_count = 0     # 当前颜色采样次数

    def add_color(self, color_name: str, lower: list, upper: list) -> None:
        """添加/更新颜色阈值"""
        self.color_thresholds[color_name] = {
            "lower": np.array(lower),
            "upper": np.array(upper),
            "count": 1,
            "last_updated": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def start_new_sample_round(self):
        """开始新一轮采样"""
        if self.current_color in self.sample_history:
            self.sample_count = len(self.sample_history[self.current_color])
        else:
            self.sample_history[self.current_color] = []
            self.sample_count = 0

    def sample_color(self, frame: np.ndarray, x: int, y: int, cap: cv2.VideoCapture) -> bool:
        h, w = frame.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return False

        # 单点采样（原逻辑）
        half_size = self.sample_size // 2
        x1, y1 = max(0, x - half_size), max(0, y - half_size)
        x2, y2 = min(w, x + half_size), min(h, y + half_size)
        
        # 多帧稳定采样
        samples = []
        for _ in range(3):
            ret, temp_frame = cap.read()
            if not ret: break
            hsv_roi = cv2.cvtColor(temp_frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
            samples.append(hsv_roi)
        
        if not samples:
            return False

        # 记录本次采样结果
        pixels = np.concatenate(samples).reshape(-1, 3)
        new_sample = {
            "min": np.min(pixels, axis=0),
            "max": np.max(pixels, axis=0),
            "position": (x, y)
        }
        
        if self.current_color not in self.sample_history:
            self.sample_history[self.current_color] = []
        self.sample_history[self.current_color].append(new_sample)
        self.sample_count += 1

        # 自动融合阈值（加权平均）
        return self._update_threshold()

    def _update_threshold(self) -> bool:
        """融合历史采样数据更新阈值"""
        if self.current_color not in self.sample_history or not self.sample_history[self.current_color]:
            return False

        samples = self.sample_history[self.current_color]
        total = len(samples)
        
        # 计算加权平均值（越新的采样权重越高）
        weights = np.linspace(0.5, 1.5, num=total)  # 权重从0.5线性增加到1.5
        weights /= weights.sum()  # 归一化

        min_vals = np.array([s["min"] for s in samples])
        max_vals = np.array([s["max"] for s in samples])
        
        weighted_min = np.sum(min_vals * weights[:, None], axis=0)
        weighted_max = np.sum(max_vals * weights[:, None], axis=0)

        # 保留20%历史阈值（如果是更新而非首次）
        if self.current_color in self.color_thresholds:
            old_min = self.color_thresholds[self.current_color]["lower"]
            old_max = self.color_thresholds[self.current_color]["upper"]
            new_min = old_min * 0.2 + weighted_min * 0.8
            new_max = old_max * 0.2 + weighted_max * 0.8
        else:
            new_min, new_max = weighted_min, weighted_max

        self.add_color(self.current_color, new_min.tolist(), new_max.tolist())
        
        print(f"{self.current_color} 阈值更新(基于{total}次采样):\n"
              f"lower={new_min.round(1)}\nupper={new_max.round(1)}\n"
              f"采样点位置: {[s['position'] for s in samples]}", flush=True)
        return True

    def detect(self, frame: np.ndarray, color_name: str) -> Optional[np.ndarray]:
        """检测特定颜色并返回二值化掩膜（结合边缘检测）"""
        if color_name not in self.color_thresholds:
            if color_name not in self._warned_colors:  # 仅首次警告
                print(f"注意: 未配置 {color_name} 的阈值，请先采样", flush=True)
                self._warned_colors.add(color_name)
            return None

        if self._hsv_cache is None or self._hsv_cache.shape != frame.shape[:2]:
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            self._hsv_cache = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        threshold = self.color_thresholds[color_name]
        mask = cv2.inRange(self._hsv_cache, threshold["lower"], threshold["upper"])
        # edges = cv2.Canny(blurred, 50, 150)  # 禁用边缘检测
        # mask = cv2.bitwise_and(mask, edges)
        return cv2.erode(mask, None, iterations=2)

    def draw_roi(self, frame: np.ndarray) -> None:
        """在图像上绘制采样区域"""
        if self.last_roi:
            x1, y1, x2, y2 = self.last_roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Sampling: {self.current_color}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def get_binary_mask(self, frame: np.ndarray, color_name: str) -> Optional[np.ndarray]:
        """生成黑白二进制图像，所选颜色为白色，其他为黑色"""
        mask = self.detect(frame, color_name)
        if mask is not None:
            return mask
        return None

    def reset_color_threshold(self, color_name: str) -> None:
        """重置指定颜色的阈值（设置为不匹配任何颜色）"""
        if color_name in self.color_thresholds:
            self.color_thresholds[color_name] = {
                "lower": np.array([255, 255, 255]),  # 设置为不匹配任何颜色
                "upper": np.array([0, 0, 0]),        # 设置为不匹配任何颜色
                "count": 1,
                "last_updated": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            self._warned_colors.discard(color_name)  # 清除该颜色的警告记录
            print(f"{color_name} 阈值已重置！二进制图像将显示为黑色。")
            self.sample_history.pop(color_name, None)
            self.sample_count = 0

# ---------------------------- 主程序 ----------------------------
def main():
    detector = ColorDetector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头！")
        return

    # 加载默认配置
    for color in ["black", "blue", "yellow", "red"]:
        cfg = ConfigManager.load_config(color)
        if cfg:
            detector.add_color(color, cfg["lower"], cfg["upper"])
            print(f"加载 {color} 阈值: lower={cfg['lower']}, upper={cfg['upper']}")  # 调试信息

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            detector.sample_color(frame, x, y, cap)  # 传递cap参数

    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Binary Mask", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Camera", mouse_callback)

    print_help()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if len(frame.shape) == 2:  # 强制转换为BGR格式
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame = cv2.flip(frame, 1)

        binary_mask = detector.get_binary_mask(frame, detector.current_color)
        if binary_mask is not None:
            cv2.imshow("Binary Mask", binary_mask)
            cv2.waitKey(1)  # 强制刷新窗口

        # 显示其他内容
        frame_copy = frame.copy()
        detector.draw_roi(frame_copy)
        cv2.imshow("Camera", frame_copy)

        key = cv2.waitKey(50) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('h'):
            print_help()
        elif ord('1') <= key <= ord('5'):
            colors = ["black", "blue", "yellow", "red", "green"]
            detector.current_color = colors[key - ord('1')]
            print(f"当前操作颜色: {detector.current_color}")
        elif key == ord('+'):
            detector.sample_size = min(100, detector.sample_size + 5)
            print(f"采样区域大小: {detector.sample_size}x{detector.sample_size}")
        elif key == ord('-'):
            detector.sample_size = max(5, detector.sample_size - 5)
            print(f"采样区域大小: {detector.sample_size}x{detector.sample_size}")
        elif key == ord('s'):
            if detector.current_color in detector.color_thresholds:
                if ConfigManager.save_config(detector.current_color, 
                                           detector.color_thresholds[detector.current_color]):
                    print(f"{detector.current_color} 配置已保存！")
        elif key == ord('r'):  # 重置当前颜色的阈值
            detector.reset_color_threshold(detector.current_color)
        elif key == ord('n'):  # 开始新一轮采样
            detector.start_new_sample_round()
            print(f"开始新一轮 {detector.current_color} 采样，请点击目标区域...")

    cap.release()
    cv2.destroyAllWindows()

def print_help():
    print("\n------ 使用说明 ------")
    print("鼠标操作:")
    print(" - 左键点击: 采样当前位置颜色")
    print("键盘命令:")
    print(" - 1-5: 选择颜色 (1=黑, 2=蓝, 3=黄, 4=红, 5=绿)")
    print(" - +/-: 调整采样区域大小")
    print(" - s: 保存当前颜色配置")
    print(" - r: 重置当前颜色阈值（二进制图像变为黑色）")
    print(" - h: 显示帮助信息")
    print(" - ESC: 退出程序")
    print("----------------------\n")

if __name__ == "__main__":
    main()