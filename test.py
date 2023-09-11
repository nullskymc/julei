import cv2
import os

# 读取图片
img = cv2.imread('./dataset/Top-0001_orign.bmp')

# 将图片转为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 对灰度图进行Otsu阈值处理
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 根据各种矿物的阈值范围进行分类
shiying = cv2.inRange(gray, 101, 200)
changshi = cv2.inRange(gray, 61, 100)
fangjie = cv2.inRange(gray, 221, 255)
niantu = cv2.inRange(gray, 41, 60)
kongxi = cv2.inRange(gray, 0, 40)

# 使用不同颜色输出分类的结果
shiying = cv2.bitwise_and(img, img, mask=shiying)
changshi = cv2.bitwise_and(img, img, mask=changshi)
fangjie = cv2.bitwise_and(img, img, mask=fangjie)
niantu = cv2.bitwise_and(img, img, mask=niantu)
kongxi = cv2.bitwise_and(img, img, mask=kongxi)

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