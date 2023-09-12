import cv2
import os

# 读取图片
img = cv2.imread('./dataset/Top-0001_orign.bmp')

# 将图片转为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 对灰度图进行Otsu阈值处理
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 将 thresh 转换为浮点数类型

# 根据各种矿物的阈值范围进行分类
shiying = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
changshi = cv2.threshold(gray, 100, 200, cv2.THRESH_BINARY)[1]
niantu = cv2.threshold(gray, 120, 200, cv2.THRESH_BINARY)[1]
fangjie = cv2.threshold(gray, 60, 100, cv2.THRESH_BINARY)[1]
kongxi = cv2.threshold(gray, 0, 60, cv2.THRESH_BINARY)[1]

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
a
#将分类结果叠加到原图上
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