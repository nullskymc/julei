import cv2

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

# 在一张图片上保存所有分类的结果
cv2.imwrite('./dataset/Top-0001_shiying.bmp', shiying)
cv2.imwrite('./dataset/Top-0001_changshi.bmp', changshi)
cv2.imwrite('./dataset/Top-0001_fangjie.bmp', fangjie)
cv2.imwrite('./dataset/Top-0001_niantu.bmp', niantu)
cv2.imwrite('./dataset/Top-0001_kongxi.bmp', kongxi)