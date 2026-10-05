import cv2 as cv
import numpy as np
import serial
from sklearn.cluster import KMeans
ser = serial.Serial("/dev/ttyAMA0", 9600)
if ser.isOpen == False:
    ser.open()
cap = cv.VideoCapture(0)
width = cap.get(3)
height = cap.get(4)
fps = cap.get(5)
ret = 1
centers=[]
count=0
count2=0
while ret:
    ret,img = cap.read()
    img = cv.flip(img,1)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)#BGR转HSV
    low_hsv = np.array([26,80,100])#这里要根据HSV表对应
    high_hsv = np.array([100,255,255])#这里填入三个max值 黄色

    mask = cv.inRange(hsv,lowerb=low_hsv,upperb=high_hsv)#提取掩膜

    #黑色背景转透明部分
    mask_contrary = mask.copy()
    mask_contrary[mask_contrary==0]=1
    mask_contrary[mask_contrary==255]=0#把黑色背景转白色
    mask_bool = mask_contrary.astype(bool)
    mask_img = cv.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
    #这个是把掩模图和原图进行叠加，获得原图上掩模图位置的区域


    GrayImage=cv.cvtColor(mask_img,cv.COLOR_BGR2GRAY)
    GrayImage= cv.medianBlur(GrayImage,5)
    th2 = cv.adaptiveThreshold(GrayImage,255,cv.ADAPTIVE_THRESH_MEAN_C,  
                        cv.THRESH_BINARY,3,5)
    
    circles = cv.HoughCircles(th2,cv.HOUGH_GRADIENT,1,40,
                                param1=50,param2=10,minRadius=10,maxRadius=80)
    try:
        c = np.uint16(np.around(circles))
        # draw the outer circle
        cv.circle(img,(c[0,0,0],c[0,0,1]),c[0,0,2],(0,255,0),2)
        # draw the center of the circle
        cv.circle(img,(c[0,0,0],c[0,0,1]),2,(0,0,255),12)
        centers.append(c[0,0])
        count=count+1
        


        if count==12:
            count=0
            kmeans = KMeans(n_clusters=1,random_state=9).fit(centers)
            del centers[:]
            center = kmeans.cluster_centers_
            c = center[0]
            print(c)
            if c[0] > 220 and c[0] < 340 and c[1]>405:
                count2=count2+1
                if count2==3:
                    count2=0
                    ser.write(b'80')
                print("tiqiu")
            if c[0]<220:
                ser.write(b'40')
                print("turn left")
            if c[0]>340:
                ser.write(b'30')
                print("turn right")
            if c[0] > 220 and c[0] < 340:
                ser.write(b'19')
                print("qianjin")

    except:
        pass
    cv.namedWindow("detected circles",cv.WINDOW_NORMAL)
    cv.imshow('detected circles',img)
    if cv.waitKey(10)==27:
        cv.destroyAllWindows()
        break

