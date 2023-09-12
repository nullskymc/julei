import cv2
import numpy as np

# 加载图像
img = cv2.imread('./dataset/Top-0001_orign.bmp')

# 将图像转换为灰度图像
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 对图像进行降噪
denoised_img = cv2.fastNlMeansDenoising(gray_img, None, 10, 7, 21)

# 将图像转换为一维数组
data = denoised_img.reshape((-1, 1))

# 进行K-Means聚类
k = 5
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
_, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 将聚类结果转换为图像
segmented_img = labels.reshape(denoised_img.shape)

# 根据聚类结果对图像进行分类
shiying = np.zeros_like(segmented_img)
changshi = np.zeros_like(segmented_img)
fangjie = np.zeros_like(segmented_img)
niantu = np.zeros_like(segmented_img)
kongxi = np.zeros_like(segmented_img)

for i in range(k):
    mask = segmented_img == i
    if np.mean(gray_img[mask]) > 200:
        shiying[mask] = 255
    elif np.mean(gray_img[mask]) > 100:
        changshi[mask] = 255
    elif np.mean(gray_img[mask]) > 60:
        fangjie[mask] = 255
    elif np.mean(gray_img[mask]) > 40:
        niantu[mask] = 255
    else:
        kongxi[mask] = 255

# 将分类结果转换为彩色图像
shiying = cv2.cvtColor(shiying, cv2.COLOR_GRAY2BGR)
changshi = cv2.cvtColor(changshi, cv2.COLOR_GRAY2BGR)
fangjie = cv2.cvtColor(fangjie, cv2.COLOR_GRAY2BGR)
niantu = cv2.cvtColor(niantu, cv2.COLOR_GRAY2BGR)
kongxi = cv2.cvtColor(kongxi, cv2.COLOR_GRAY2BGR)

# 使用不同颜色输出分类的结果
#石英为红色
shiying[:, :, 0] = 0
shiying[:, :, 1] = 0
#长石为绿色
changshi[:, :, 0] = 0
changshi[:, :, 2] = 0
#方解石为蓝色
fangjie[:, :, 1] = 0
fangjie[:, :, 2] = 0
#粘土为玫红色
niantu[:, :, 0] = 152
niantu[:, :, 1] = 125

# 将分类结果叠加到原图上
img = cv2.add(img, shiying)
img = cv2.add(img, changshi)
img = cv2.add(img, fangjie)
img = cv2.add(img, niantu)
img = cv2.add(img, kongxi)

# 创建保存分类结果的文件夹
if not os.path.exists('./dataset/shiying'):
    os.makedirs('./dataset/shiying')
if not os.path.exists('./dataset/changshi'):
    os.makedirs('./dataset/changshi')
if not os.path.exists('./dataset/fangjie'):
    os.makedirs('./dataset/fangjie')
if not os.path.exists('./dataset/niantu'):
    os.makedirs('./dataset/niantu')
if not os.path.exists('./dataset/kongxi'):
    os.makedirs('./dataset/kongxi')

# 保存分类结果图片到对应的文件夹
cv2.imwrite('./dataset/shiying/Top-0001_shiying.bmp', shiying)
cv2.imwrite('./dataset/changshi/Top-0001_changshi.bmp', changshi)
cv2.imwrite('./dataset/fangjie/Top-0001_fangjie.bmp', fangjie)
cv2.imwrite('./dataset/niantu/Top-0001_niantu.bmp', niantu)
cv2.imwrite('./dataset/kongxi/Top-0001_kongxi.bmp', kongxi)
cv2.imwrite('./dataset/Top-0001_result.bmp', img)