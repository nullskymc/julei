import cv2
import numpy as np

# 加载图像
img = cv2.imread('./dataset/Top-0001_orign.bmp')

# 将图像转换为灰度图像
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 对图像进行降噪
denoised_img = cv2.fastNlMeansDenoising(gray_img, None, 10, 7, 21)

#保存降噪后的图像
cv2.imwrite('denoised.png', denoised_img)
