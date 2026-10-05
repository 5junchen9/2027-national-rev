class BaseDataForRobotHead:
    #headVerticalError = -8  # 头部舵机垂直(上下)误差校正值 调节该参数 保证90度头部摄像头直视前方
    headVerticalError = -8  # 头部舵机垂直(上下)误差校正值 调节该参数 保证90度头部摄像头直视前方 
    headHorizontalError = 0  # 头部舵机水平(左右)误差校正值 90度是正前方


class BaseData:
    class FindRectangle:
        findTimes = 1

    class Move:
        Turn_90 = 4  # 机器人旋转90度需要转动几步

    class Image:
        width = 640
        height = 480
        half_Width = 320
        half_Height = 240

    class Barcode:
        findBarcodeLoopCount = 1  # 摄像头检测条码有效的循环次数

        class HeadAnger:
            near = 106  # 看近处
            far = 100  # 看远处
            #near = 30  # 看近处
            #far = 40  # 看远处
    class FindBox:
        findBoxLoopCount = 6  # 摄像头检测箱子有效的循环次数
        findBoxCount = 4  # 摄像头检测到多少次箱子认为有效

        nearRightLimit = 200  # 需要右转的限值
        nearLeftLimit = 450  # 需要左转的限值
        boxNearLimit = 380  # 取盒子距离限值

        class HeadAnger:
            class FindBox:
                #near = 30  # 脚下
                #far = 40  # 看远处
                near = 106  # 脚下
                far = 100  # 看远处
            class FindBarcode:
                #near = 30  # 看近处
                #far = 40  # 看远处
                near = 106  # 看近处
                far = 100  # 看远处