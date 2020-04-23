# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:15:45 2020

@author: admin
"""

import cv2

#放缩变换
img = cv2.imread("C:/Users/admin/Desktop/cv/task02/1.jpg")
#下面的None本应该是输出图像的尺寸，但是因为后面我们设置了缩放因子，所以，这里为None
res = cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
cv2.imshow("res",res)
#这里直接设置输出图像的尺寸，所以不用设置缩放因子
height , width =img.shape[:2]
res = cv2.resize(img,(2*width,2*height),interpolation=cv2.INTER_CUBIC)
cv2.imshow("res",res)
cv2.waitKey(0)


#平移变换
import cv2
import numpy as np
img = cv2.imread("C:/Users/admin/Desktop/cv/task02/1.jpg")
rows,cols,channel=img.shape
#这里的第一个参数为旋转中心，第二个为旋转角度，第三个为旋转后的缩放因子
#可以通过设置旋转中心，缩放因子以及窗口大小来防止旋转后超出边界的问题。
M=np.float32([[1,0,100],[0,1,50]])
#第三个参数是输出图像的尺寸中心
dest=cv2.warpAffine(img,M,(cols,rows))
cv2.imshow('img',dest)
cv2.waitKey(0)
cv2.destroyAllWindows()

#旋转变换
import cv2
import numpy as np
img = cv2.imread("C:/Users/admin/Desktop/cv/task02/1.jpg")
rows,cols,channel=img.shape
#这里的第一个参数为旋转中心，第二个为旋转角度，第三个为旋转后的缩放因子
#可以通过设置旋转中心，缩放因子以及窗口大小来防止旋转后超出边界的问题。
M=cv2.getRotationMatrix2D((cols/2,rows/3),90,0.4)
#第三个参数是输出图像的尺寸中心
dst=cv2.warpAffine(img,M,(cols,rows))
cv2.imshow('img',dest)
cv2.waitKey(0)
cv2.destroyAllWindows()


#仿射变换
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("C:/Users/admin/Desktop/cv/task02/1.jpg")
rows,cols,channel=img.shape

pts1 = np.float32([50,50],[200,50],[50,200])
pts2 = np.float32([10,100],[200,50],[100,250])
#行，列，通道数
M = cv2.getAffineTransform(pts1,pts2)
dst = cv2.warpAffine(img,M,(cols,rows))

plt.subplot(121,plt.imshow(img),plt.title('Input'))
plt.subplot(122,plt.imshow(img),plt.title('output'))
plt.show()

