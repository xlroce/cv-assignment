import cv2
import random
import numpy as np
from matplotlib import pyplot as plt


class CombineImageClass:

    def __init__(self, path):
        # read image and save in class
        self.img = cv2.imread(path)

    '''
        image crop 
    '''

    def crop_change(self, x, y, w, h):
        shape = self.img.shape
        if x < shape[0] and y < shape[1]:
            crop_img = self.img[y: h + y, x: w + x]
            cv2.imshow("show image", crop_img)
        else:
            print("incorrect image size")
    '''
        exit image with esc
    '''

    def color_shift(self):
        img = self.img
        # 分别对RGB三色进行处理
        for i in range(3):
            ran = random.randint(-50, 50)
            print(ran)
            if ran > 0:
                lim = 255 - ran
                img[i][img[i] > lim] = 255
                img[i][img[i] <= lim] = (img[i][img[i] <= lim] + ran).astype(img.dtype)
            else:
                lim = -ran
                img[i][img[i] < lim] = 0
                img[i][img[i] >= lim] = (img[i][img[i] >= lim] + ran).astype(img.dtype)
        cv2.imshow("show image", img)

    def rotation(self, angle, scale):
        img = self.img
        M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, scale)  # center, angle, scale
        img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        cv2.imshow('rotated lenna', img_rotate)

    def perspective_transform(self):
        img = self.img
        (height, width, channels) = img.shape

        # warp:
        random_margin = 60
        x1 = random.randint(-random_margin, random_margin)
        y1 = random.randint(-random_margin, random_margin)
        x2 = random.randint(width - random_margin - 1, width - 1)
        y2 = random.randint(-random_margin, random_margin)
        x3 = random.randint(width - random_margin - 1, width - 1)
        y3 = random.randint(height - random_margin - 1, height - 1)
        x4 = random.randint(-random_margin, random_margin)
        y4 = random.randint(height - random_margin - 1, height - 1)

        dx1 = random.randint(-random_margin, random_margin)
        dy1 = random.randint(-random_margin, random_margin)
        dx2 = random.randint(width - random_margin - 1, width - 1)
        dy2 = random.randint(-random_margin, random_margin)
        dx3 = random.randint(width - random_margin - 1, width - 1)
        dy3 = random.randint(height - random_margin - 1, height - 1)
        dx4 = random.randint(-random_margin, random_margin)
        dy4 = random.randint(height - random_margin - 1, height - 1)

        pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
        M_warp = cv2.getPerspectiveTransform(pts1, pts2)
        img_warp = cv2.warpPerspective(img, M_warp, (width, height))
        cv2.imshow("warp", img_warp)
    def exit(self):
        key = cv2.waitKey()
        if key == 27:
            cv2.destroyAllWindows()


com = CombineImageClass("img.jpg")
# com.crop_change(200, 0, 200, 512)
# com.color_shift()
# 角度, 缩放比例
# com.rotation(30, 0.5)
com.perspective_transform()
com.exit()

