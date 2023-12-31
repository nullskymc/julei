import cv2
import numpy as np
from sklearn.cluster import KMeans

# 读取图像
img = cv2.imread('./dataset/Top-0001.bmp')

# 将图像转换为灰度图像
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 对灰度图像进行预处理
denoised_img = cv2.fastNlMeansDenoising(gray_img, None, 10, 7, 21)

# 将图像转换为一维数组
data = denoised_img.reshape(-1, 1)

# 使用KMeans算法进行聚类
kmeans = KMeans(n_clusters=5, random_state=0, n_init=10).fit(data)
labels = kmeans.predict(data)

# 将聚类结果转换为图像
segmented_img = labels.reshape(denoised_img.shape)

# 根据聚类结果对图像进行分类
fangjie = np.zeros_like(segmented_img)
shiying = np.zeros_like(segmented_img)
niantu = np.zeros_like(segmented_img)
kongxi = np.zeros_like(segmented_img)
changshi = np.zeros_like(segmented_img)

segmented_img = segmented_img.astype(np.uint8)
# 运算优化边界
open_mask = np.ones((3, 3), np.uint8)
segmented_img = cv2.morphologyEx(segmented_img, cv2.MORPH_OPEN, open_mask)
close_mask = np.ones((3, 3), np.uint8)
segmented_img = cv2.morphologyEx(segmented_img, cv2.MORPH_CLOSE, close_mask)

for i in range(kmeans.n_clusters):
    mask = segmented_img == i
    if i == 0:
        shiying[mask] = 255
    elif i == 1:
        kongxi[mask] = 255
    elif i == 2:
        niantu[mask] = 255
    elif i == 3:
        changshi[mask] = 255
    else:
        fangjie[mask] = 255

# 将分类结果转换为彩色图像
fangjie = cv2.cvtColor(fangjie.astype(np.uint8), cv2.COLOR_GRAY2BGR)
shiying = cv2.cvtColor(shiying.astype(np.uint8), cv2.COLOR_GRAY2BGR)
niantu = cv2.cvtColor(niantu.astype(np.uint8), cv2.COLOR_GRAY2BGR)
kongxi = cv2.cvtColor(kongxi.astype(np.uint8), cv2.COLOR_GRAY2BGR)
changshi = cv2.cvtColor(changshi.astype(np.uint8), cv2.COLOR_GRAY2BGR)

#对结果上色
fangjie[:, :, 0] = 0

shiying[:, :, 0] = 0

niantu[:, :, 0] = 0
niantu[:, :, 1] = 0

kongxi[:, :, 0] = 0
kongxi[:, :, 1] = 0
kongxi[:, :, 2] = 0

changshi[:, :, 1] = 0
changshi[:, :, 2] = 0



# 将分类结果叠加到原图上
img = cv2.add(img, fangjie)
img = cv2.add(img, shiying)
img = cv2.add(img, niantu)
img = cv2.add(img, kongxi)
img = cv2.add(img, changshi)


#创建保存文件夹
import os
if not os.path.exists('./output'):
    os.mkdir('./output')
if not os.path.exists('./output/fangjie'):
    os.mkdir('./output/fangjie')
if not os.path.exists('./output/shiying'):
    os.mkdir('./output/shiying')
if not os.path.exists('./output/niantu'):
    os.mkdir('./output/niantu')
if not os.path.exists('./output/kongxi'):
    os.mkdir('./output/kongxi')
if not os.path.exists('./output/changshi'):
    os.mkdir('./output/changshi')

# 保存分类结果图片到对应的文件夹
cv2.imwrite('./output/fangjie/Top-0001_fangjie.bmp', fangjie)
cv2.imwrite('./output/shiying/Top-0001_shiying.bmp', shiying)
cv2.imwrite('./output/niantu/Top-0001_niantu.bmp', niantu)
cv2.imwrite('./output/kongxi/Top-0001_kongxi.bmp', kongxi)
cv2.imwrite('./output/changshi/Top-0001_changshi.bmp', changshi)
cv2.imwrite('./output/Top-0001_result.bmp', img)