import datetime
import json
import os
import time
import numpy as np
import cv2
from typing import Dict, Optional, Tuple
from robot_config import ROOT
from roboteye import RobotEye

# ---------------------------- 颜色检测核心类 ----------------------------
class ColorDetector:
    def __init__(self):
        self.color_thresholds: Dict[str, Dict] = {}  # 存储多颜色阈值
        self.sample_size = 20  # 默认采样区域大小
        self.current_color = "black"  # 当前操作的颜色
        self.last_roi = None  # 最后采样的区域坐标
        self._hsv_cache = None
        self._warned_colors = set()  # 记录已警告的颜色
        
        # 设置配置文件保存路径
        self.config_dir = str(ROOT / "assets/hsv")
        
        # 初始化默认颜色阈值
        self.default_colors = {
            "black": {"lower": [0, 0, 0], "upper": [180, 255, 30]},
            "blue": {"lower": [100, 50, 50], "upper": [130, 255, 255]},
            "yellow": {"lower": [20, 100, 100], "upper": [30, 255, 255]},
            "red": {"lower": [0, 100, 100], "upper": [10, 255, 255]},
            "green": {"lower": [40, 50, 50], "upper": [80, 255, 255]},
            "ball": {"lower": [0, 0, 0], "upper": [255, 255, 255]},
        }
        
        # 加载默认颜色阈值
        for color in self.default_colors:
            self.add_color(color, 
                          self.default_colors[color]["lower"],
                          self.default_colors[color]["upper"])
        
        # 尝试从文件加载保存的配置
        self.load_saved_configs()

    def load_saved_configs(self):
        """从文件加载保存的颜色配置"""
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
            
        for color in self.default_colors.keys():
            config_file = os.path.join(self.config_dir, f"{color}hsv.json")
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r') as f:
                        data = json.load(f)
                        self.add_color(color, data["lower"], data["upper"])
                        print(f"已加载 {color} 的保存配置")
                except Exception as e:
                    print(f"加载 {color} 配置失败: {e}")

    def save_current_config(self):
        """保存当前颜色的配置到文件"""
        if self.current_color not in self.color_thresholds:
            return False
            
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
            
        config_file = os.path.join(self.config_dir, f"{self.current_color}hsv.json")
        try:
            data = {
                "lower": self.color_thresholds[self.current_color]["lower"].tolist(),
                "upper": self.color_thresholds[self.current_color]["upper"].tolist(),
                "last_updated": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(config_file, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"{self.current_color} 配置已保存到文件: {config_file}")
            return True
        except Exception as e:
            print(f"保存 {self.current_color} 配置失败: {e}")
            return False

    def add_color(self, color_name: str, lower: list, upper: list) -> None:
        """添加/更新颜色阈值"""
        self.color_thresholds[color_name] = {
            "lower": np.array(lower),
            "upper": np.array(upper),
            "count": 1,
            "last_updated": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def sample_color(self, frame: np.ndarray, x: int, y: int, cap: RobotEye) -> bool:
        """从指定位置采样颜色（多帧平均）"""
        h, w = frame.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            print("错误：采样位置超出图像范围！")
            return False

        # 动态调整采样区域大小
        half_size = max(5, min(self.sample_size // 2, 50))
        x1, y1 = max(0, x - half_size), max(0, y - half_size)
        x2, y2 = min(w, x + half_size), min(h, y + half_size)
        self.last_roi = (x1, y1, x2, y2)

        # 多帧采样平均
        samples = []
        for _ in range(3):
            ret, temp_frame = cap.getImage()
            if not ret:
                break
            hsv_roi = cv2.cvtColor(temp_frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
            samples.append(hsv_roi)

        if not samples:
            return False

        # 计算平均最小/最大HSV值
        avg_pixels = np.concatenate(samples).reshape(-1, 3)
        min_hsv = np.min(avg_pixels, axis=0)
        max_hsv = np.max(avg_pixels, axis=0)

        # 更新阈值
        self.add_color(self.current_color, min_hsv.tolist(), max_hsv.tolist())
        print(f"{self.current_color} 阈值更新: lower={min_hsv}, upper={max_hsv}")
        return True

    def detect(self, frame: np.ndarray, color_name: str) -> Optional[np.ndarray]:
        """检测特定颜色并返回二值化掩膜"""
        if color_name not in self.color_thresholds:
            if color_name not in self._warned_colors:
                print(f"注意: 未配置 {color_name} 的阈值，使用默认值", flush=True)
                self._warned_colors.add(color_name)
            return None

        if self._hsv_cache is None or self._hsv_cache.shape != frame.shape[:2]:
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            self._hsv_cache = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        threshold = self.color_thresholds[color_name]
        mask = cv2.inRange(self._hsv_cache, threshold["lower"], threshold["upper"])
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
        """重置指定颜色的阈值为默认值"""
        if color_name in self.default_colors:
            self.add_color(color_name,
                          self.default_colors[color_name]["lower"],
                          self.default_colors[color_name]["upper"])
            self._warned_colors.discard(color_name)
            print(f"{color_name} 阈值已重置为默认值！")

# ---------------------------- 主程序 ----------------------------
def main():
    detector = ColorDetector()
    cap = RobotEye()

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            detector.sample_color(frame, x, y, cap)

    try:
        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Binary Mask", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Camera", mouse_callback)
        print_help()
        while True:
            ret, frame = cap.getImage()
            if not ret:
                break
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            binary_mask = detector.get_binary_mask(frame, detector.current_color)
            if binary_mask is not None:
                cv2.imshow("Binary Mask", binary_mask)
            frame_copy = frame.copy()
            detector.draw_roi(frame_copy)
            cv2.imshow("Camera", frame_copy)
            key = cv2.waitKey(50) & 0xFF
            if key == 27:
                break
            elif key == ord('h'):
                print_help()
            elif ord('1') <= key <= ord('6'):
                colors = ["black", "blue", "yellow", "red", "green", "ball"]
                detector.current_color = colors[key - ord('1')]
                print(f"当前操作颜色: {detector.current_color}")
            elif key == ord('+'):
                detector.sample_size = min(100, detector.sample_size + 5)
            elif key == ord('-'):
                detector.sample_size = max(5, detector.sample_size - 5)
            elif key == ord('s'):
                detector.save_current_config()
            elif key == ord('r'):
                detector.reset_color_threshold(detector.current_color)
    finally:
        cap.close()
        cv2.destroyAllWindows()

def print_help():
    print("\n------ 使用说明 ------")
    print("鼠标操作:")
    print(" - 左键点击: 采样当前位置颜色")
    print("键盘命令:")
    print(" - 1-5: 选择颜色 (1=黑, 2=蓝, 3=黄, 4=红, 5=绿, 6=球)")
    print(" - +/-: 调整采样区域大小")
    print(" - s: 保存当前颜色配置到文件")
    print(" - r: 重置当前颜色阈值为默认值")
    print(" - h: 显示帮助信息")
    print(" - ESC: 退出程序")
    print("----------------------\n")

if __name__ == "__main__":
    main()
