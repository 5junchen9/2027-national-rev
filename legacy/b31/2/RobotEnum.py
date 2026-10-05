from enum import Enum


# 抱盒子流程状态
class BoxProcessState(Enum):
    NONE = 0,
    FIND_BOX = 10,
    UP_BOX = 20,
    DOWN_BOX = 30,
    BARCODE = 40,
    MOVE_UP = 50,  # 前进
    MOVE_LEFT = 60,  # 右转
    FIND_BALL = 70,
    FLAG = 0,