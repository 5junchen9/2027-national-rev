import math


class MyAngle:

    def __azimuthangle(self, x1, y1, x2, y2):
        """ 已知两点坐标计算角度 -
        :param x1: 原点横坐标值
        :param y1: 原点纵坐标值
        :param x2: 目标点横坐标值
        :param y2: 目标纵坐标值
        """
        dx = x2 - x1
        dy = y2 - y1
        # 求斜率
        k = dy / dx
        # 结果是弧度值
        angle = math.atan(k)
        # 弧度值转为角度
        return angle * 180 / math.pi

    def get_angle(self, x1, y1, x2, y2):
        """
        获取出进行矫正所需要的角度
        """
        return self.__azimuthangle(x1, y1, x2, y2)
        # 将坐标从下到上，从左到右进行排序
        # locs = {x1, y1, x2, y2}
        # locs = sorted(locs, key=lambda x: x.y * 100000 + x.x * 1000)
        # return self.__azimuthangle(locs[2].x, locs[2].y, locs[3].x, locs[3].y)
