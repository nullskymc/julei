import cv2
import numpy as np

# 加载图像
img = cv2.imread('./dataset/Top-0001.bmp')

# 将图像转换为灰度图像
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 对图像进行降噪
denoised_img = cv2.fastNlMeansDenoising(gray_img, None, 10, 7, 21)

# 将图像转换为一维数组，并将数据类型转换为np.float32
data = denoised_img.reshape((-1, 1)).astype(np.float32)

# 进行K-Means聚类
k = 5
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
_, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 将聚类结果转换为图像
segmented_img = labels.reshape(denoised_img.shape)

# 根据聚类结果对图像进行分类
fangjie = np.zeros_like(segmented_img)
shiying = np.zeros_like(segmented_img)
niantu = np.zeros_like(segmented_img)
kongxi = np.zeros_like(segmented_img)
changshi = np.zeros_like(segmented_img)

for i in range(k):
    mask = segmented_img == i
    if np.mean(gray_img[mask]) > 220:
        fangjie[mask] = 255
    elif np.mean(gray_img[mask]) > 100:
        shiying[mask] = 255
    elif np.mean(gray_img[mask]) > 85:
        changshi[mask] = 255
    elif np.mean(gray_img[mask]) > 46:
        niantu[mask] = 255
    else:
        kongxi[mask] = 255

# 将分类结果转换为彩色图像
fangjie = cv2.cvtColor(fangjie.astype(np.uint8), cv2.COLOR_GRAY2BGR)
shiying = cv2.cvtColor(shiying.astype(np.uint8), cv2.COLOR_GRAY2BGR)
niantu = cv2.cvtColor(niantu.astype(np.uint8), cv2.COLOR_GRAY2BGR)
kongxi = cv2.cvtColor(kongxi.astype(np.uint8), cv2.COLOR_GRAY2BGR)
changshi = cv2.cvtColor(changshi.astype(np.uint8), cv2.COLOR_GRAY2BGR)

# 使用不同颜色输出分类的结果
#石英为红色
fangjie[:, :, 0] = 0
fangjie[:, :, 1] = 0
#长石为绿色
shiying[:, :, 0] = 0
shiying[:, :, 2] = 0
#方解石为蓝色
fangjie[:, :, 1] = 0
fangjie[:, :, 2] = 0

# 将分类结果叠加到原图上
img = cv2.add(img, fangjie)
img = cv2.add(img, shiying)
img = cv2.add(img, niantu)
img = cv2.add(img, kongxi)
img = cv2.add(img, changshi)

# 保存分类结果图片到对应的文件夹
cv2.imwrite('./dataset/fangjie/Top-0001_fangjie.bmp', fangjie)
cv2.imwrite('./dataset/shiying/Top-0001_shiying.bmp', shiying)
cv2.imwrite('./dataset/niantu/Top-0001_niantu.bmp', niantu)
cv2.imwrite('./dataset/kongxi/Top-0001_kongxi.bmp', kongxi)
cv2.imwrite('./dataset/changshi/Top-0001_changshi.bmp', changshi)
cv2.imwrite('./dataset/Top-0001_result.bmp', img)