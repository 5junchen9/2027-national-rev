import os
import time
import datetime
import threading
import queue
import sys
from pathlib import Path
import cv2
import numpy as np
from collections import deque

# 与当前项目共用中文播报，避免默认英文发音器逐字念 Chinese letter。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chinese_speech import speak_chinese

# GPIO控制
if os.sep == "/":
    import RPi.GPIO as GPIO

class RobotMovement:
    """机器人移动控制类"""
    # 定义GPIO引脚
    LEFT_MOTOR_FORWARD = 17   # 左电机前进
    LEFT_MOTOR_BACKWARD = 18  # 左电机后退
    RIGHT_MOTOR_FORWARD = 22  # 右电机前进
    RIGHT_MOTOR_BACKWARD = 23 # 右电机后退
    
    # PWM引脚用于控制速度
    LEFT_MOTOR_PWM = 12
    RIGHT_MOTOR_PWM = 13
    
    def __init__(self):
        self.__initGPIO()
        self.speed = 50  # 默认速度50%

    def __initGPIO(self):
        """初始化GPIO引脚"""
        if os.sep == "/":
            GPIO.cleanup()
            GPIO.setmode(GPIO.BCM)
            
            # 设置电机控制引脚为输出
            GPIO.setup(self.LEFT_MOTOR_FORWARD, GPIO.OUT)
            GPIO.setup(self.LEFT_MOTOR_BACKWARD, GPIO.OUT)
            GPIO.setup(self.RIGHT_MOTOR_FORWARD, GPIO.OUT)
            GPIO.setup(self.RIGHT_MOTOR_BACKWARD, GPIO.OUT)
            
            # 设置PWM引脚
            GPIO.setup(self.LEFT_MOTOR_PWM, GPIO.OUT)
            GPIO.setup(self.RIGHT_MOTOR_PWM, GPIO.OUT)
            
            # 初始化PWM
            self.left_pwm = GPIO.PWM(self.LEFT_MOTOR_PWM, 1000)
            self.right_pwm = GPIO.PWM(self.RIGHT_MOTOR_PWM, 1000)
            self.left_pwm.start(0)
            self.right_pwm.start(0)
            
            # 初始状态停止
            self.stop()

    def __setMotorSpeed(self, left_speed, right_speed):
        """设置左右电机速度"""
        if os.sep == "/":
            self.left_pwm.ChangeDutyCycle(left_speed)
            self.right_pwm.ChangeDutyCycle(right_speed)

    def __setMotorDirection(self, left_forward, left_backward, right_forward, right_backward):
        """设置电机方向"""
        if os.sep == "/":
            GPIO.output(self.LEFT_MOTOR_FORWARD, left_forward)
            GPIO.output(self.LEFT_MOTOR_BACKWARD, left_backward)
            GPIO.output(self.RIGHT_MOTOR_FORWARD, right_forward)
            GPIO.output(self.RIGHT_MOTOR_BACKWARD, right_backward)

    def forward(self, speed=None):
        """前进"""
        if speed is None:
            speed = self.speed
        print(f"机器人前进，速度: {speed}%")
        self.__setMotorDirection(True, False, True, False)
        self.__setMotorSpeed(speed, speed)

    def backward(self, speed=None):
        """后退"""
        if speed is None:
            speed = self.speed
        print(f"机器人后退，速度: {speed}%")
        self.__setMotorDirection(False, True, False, True)
        self.__setMotorSpeed(speed, speed)

    def turn_left(self, speed=None):
        """左转"""
        if speed is None:
            speed = self.speed
        print(f"机器人左转，速度: {speed}%")
        self.__setMotorDirection(False, True, True, False)
        self.__setMotorSpeed(speed, speed)

    def turn_right(self, speed=None):
        """右转"""
        if speed is None:
            speed = self.speed
        print(f"机器人右转，速度: {speed}%")
        self.__setMotorDirection(True, False, False, True)
        self.__setMotorSpeed(speed, speed)

    def stop(self):
        """停止"""
        print("机器人停止")
        self.__setMotorDirection(False, False, False, False)
        self.__setMotorSpeed(0, 0)

    def set_speed(self, speed):
        """设置速度 (0-100)"""
        assert 0 <= speed <= 100, "速度必须在0-100之间"
        self.speed = speed
        print(f"设置速度为: {speed}%")

    def cleanup(self):
        """清理GPIO资源"""
        if os.sep == "/":
            self.stop()
            self.left_pwm.stop()
            self.right_pwm.stop()
            GPIO.cleanup()

class FaceRecognition:
    """人脸识别类"""
    def __init__(self):
        # 加载人脸检测器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 已知人脸数据库 (简单示例)
        self.known_faces = {
            "张三": "这是张三，我的朋友",
            "李四": "这是李四，我的同事",
            "王五": "这是王五，我的家人"
        }
        
        # 人脸识别历史记录
        self.face_history = deque(maxlen=10)
        self.last_recognition_time = 0
        self.recognition_cooldown = 3  # 3秒冷却时间

    def detect_faces(self, frame):
        """检测人脸"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return faces

    def recognize_face(self, frame, faces):
        """识别人脸并返回识别结果"""
        current_time = time.time()
        
        # 检查冷却时间
        if current_time - self.last_recognition_time < self.recognition_cooldown:
            return None
        
        if len(faces) > 0:
            # 简单的人脸识别逻辑（实际应用中可以使用更复杂的算法）
            # 这里使用人脸数量作为识别依据
            face_count = len(faces)
            
            # 模拟识别结果
            if face_count == 1:
                # 随机选择一个已知人脸
                import random
                person_name = random.choice(list(self.known_faces.keys()))
                description = self.known_faces[person_name]
                
                self.last_recognition_time = current_time
                return {
                    "name": person_name,
                    "description": description,
                    "face_count": face_count
                }
            elif face_count > 1:
                self.last_recognition_time = current_time
                return {
                    "name": "多人",
                    "description": f"检测到{face_count}个人",
                    "face_count": face_count
                }
        
        return None

    def draw_faces(self, frame, faces, recognition_result=None):
        """在图像上绘制人脸框和识别结果"""
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        if recognition_result:
            # 在图像上显示识别结果
            text = f"{recognition_result['name']}: {recognition_result['description']}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame

class VoiceBroadcast:
    """语音播报类"""
    def speak(self, text):
        """播报文本"""
        print(f"语音播报: {text}")
        return speak_chinese(text)

class RobotFaceRecognition:
    """机器人人脸识别主类"""
    def __init__(self):
        self.movement = RobotMovement()
        self.face_recognition = FaceRecognition()
        self.voice_broadcast = VoiceBroadcast()
        
        # 摄像头
        self.camera = None
        self.camera_thread = None
        self.is_running = False
        
        # 移动控制
        self.movement_queue = queue.Queue()
        self.movement_thread = None
        
        # 状态
        self.is_moving = False
        self.current_direction = "停止"

    def init_camera(self):
        """初始化摄像头"""
        try:
            # 尝试不同的摄像头索引
            for i in range(3):
                self.camera = cv2.VideoCapture(i)
                if self.camera.isOpened():
                    print(f"摄像头初始化成功，使用索引: {i}")
                    return True
            
            # 如果都失败，尝试默认
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                print("摄像头初始化成功，使用默认索引")
                return True
            else:
                print("摄像头初始化失败")
                return False
        except Exception as e:
            print(f"摄像头初始化错误: {e}")
            return False

    def camera_worker(self):
        """摄像头工作线程"""
        while self.is_running:
            if self.camera and self.camera.isOpened():
                ret, frame = self.camera.read()
                if ret:
                    # 检测人脸
                    faces = self.face_recognition.detect_faces(frame)
                    
                    # 识别人脸
                    recognition_result = self.face_recognition.recognize_face(frame, faces)
                    
                    # 如果识别到人脸，进行语音播报
                    if recognition_result:
                        self.voice_broadcast.speak(recognition_result['description'])
                    
                    # 绘制人脸框和识别结果
                    frame = self.face_recognition.draw_faces(frame, faces, recognition_result)
                    
                    # 显示图像
                    cv2.imshow('机器人人脸识别', frame)
                    
                    # 检查按键
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('w'):
                        self.movement_queue.put("前进")
                    elif key == ord('s'):
                        self.movement_queue.put("后退")
                    elif key == ord('a'):
                        self.movement_queue.put("左转")
                    elif key == ord('d'):
                        self.movement_queue.put("右转")
                    elif key == ord(' '):
                        self.movement_queue.put("停止")
                else:
                    print("无法读取摄像头画面")
                    break
            else:
                print("摄像头未就绪")
                break
        
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()

    def movement_worker(self):
        """移动控制工作线程"""
        while self.is_running:
            try:
                # 从队列获取移动命令
                command = self.movement_queue.get(timeout=0.1)
                
                if command == "前进":
                    self.movement.forward()
                    self.current_direction = "前进"
                elif command == "后退":
                    self.movement.backward()
                    self.current_direction = "后退"
                elif command == "左转":
                    self.movement.turn_left()
                    self.current_direction = "左转"
                elif command == "右转":
                    self.movement.turn_right()
                    self.current_direction = "右转"
                elif command == "停止":
                    self.movement.stop()
                    self.current_direction = "停止"
                
                self.is_moving = self.current_direction != "停止"
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"移动控制错误: {e}")

    def start(self):
        """启动机器人"""
        print("启动机器人人脸识别系统...")
        
        # 初始化摄像头
        if not self.init_camera():
            print("摄像头初始化失败，程序退出")
            return
        
        self.is_running = True
        
        # 启动摄像头线程
        self.camera_thread = threading.Thread(target=self.camera_worker)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        
        # 启动移动控制线程
        self.movement_thread = threading.Thread(target=self.movement_worker)
        self.movement_thread.daemon = True
        self.movement_thread.start()
        
        # 语音播报启动信息
        self.voice_broadcast.speak("机器人人脸识别系统已启动，开始巡逻")
        
        print("机器人已启动，按以下键控制移动:")
        print("W - 前进")
        print("S - 后退")
        print("A - 左转")
        print("D - 右转")
        print("空格 - 停止")
        print("Q - 退出")
        
        try:
            # 主循环
            while self.is_running:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n收到中断信号，正在停止...")
        finally:
            self.stop()

    def stop(self):
        """停止机器人"""
        print("正在停止机器人...")
        self.is_running = False
        
        # 停止移动
        self.movement.stop()
        
        # 等待线程结束
        if self.camera_thread:
            self.camera_thread.join(timeout=2)
        if self.movement_thread:
            self.movement_thread.join(timeout=2)
        
        # 清理资源
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        self.movement.cleanup()
        
        print("机器人已停止")

    def patrol_mode(self):
        """巡逻模式 - 自动移动并识别人脸"""
        print("启动巡逻模式...")
        self.voice_broadcast.speak("开始巡逻模式")
        
        # 简单的巡逻逻辑
        patrol_sequence = [
            ("前进", 3),
            ("停止", 1),
            ("左转", 1),
            ("停止", 1),
            ("前进", 3),
            ("停止", 1),
            ("右转", 1),
            ("停止", 1)
        ]
        
        for action, duration in patrol_sequence:
            if not self.is_running:
                break
            
            self.movement_queue.put(action)
            time.sleep(duration)
        
        self.movement_queue.put("停止")
        print("巡逻模式结束")

# 使用示例
if __name__ == "__main__":
    try:
        robot = RobotFaceRecognition()
        
        # 启动机器人
        robot.start()
        
    except Exception as e:
        print(f"程序运行错误: {e}")
    finally:
        print("程序结束") 
