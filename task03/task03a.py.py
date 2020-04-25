# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 20:15:14 2020

@author: admin
"""

import cv2
import numpy as np
 
cap = cv2.VideoCapture(0)
while(True):
    #ret, frame = cap.read('C:/Users/admin/Desktop/cv/task03/3.jpg')#获取图像
    frame=cv2.imread('C:/Users/admin/Desktop/cv/task03/3.jpg')
    # 转换到 HSV
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # 设定橙色的阈值
    lower_blue=np.array([0,50,50])
    upper_blue=np.array([20,255,255])
    # 根据阈值构建掩模
    mask=cv2.inRange(hsv,lower_blue,upper_blue)
    # 对原图像和掩模进行位运算
    res=cv2.bitwise_and(frame,frame,mask=mask)
    # 显示图像
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    if cv2.waitKey(1) & 0xFF == ord('q'):#监测到键盘输入q关闭
        break
cap.release()#释放摄像头
cv2.destroyAllWindows()#关闭窗口
