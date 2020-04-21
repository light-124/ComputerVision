# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 20:14:52 2020

@author: admin
"""

import cv2
if __name__ == "__main__":
    img = cv2.imread('C:/Users/admin/Desktop/1.jpg', cv2.IMREAD_UNCHANGED)
    
    print('Original Dimensions : ',img.shape) #打印图片的形状，cv里面图片的格式为(h,w,c)
    
    scale_percent = 30       # percent of original size 30/100是放缩比例
    width = int(img.shape[1] * scale_percent / 100)  #放缩后的wight
    height = int(img.shape[0] * scale_percent / 100) #放缩后的height
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR)#双线性插值缩小到0.3倍

    fx = 1.5  #放缩倍数
    fy = 1.5

    resized1 = cv2.resize(resized, dsize=None, fx=fx, fy=fy, interpolation = cv2.INTER_NEAREST)
    #用最邻近插值放缩到1.5倍数
    resized2 = cv2.resize(resized, dsize=None, fx=fx, fy=fy, interpolation = cv2.INTER_LINEAR)
    #用双线性插值放缩到1.5倍
    print('Resized Dimensions : ',resized.shape)#输出放缩后的图片大小
    
    #依次显示图片
    cv2.imshow("Resized image", resized)
    cv2.imshow("INTER_NEAREST image", resized1)
    cv2.imshow("INTER_LINEAR image", resized2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()