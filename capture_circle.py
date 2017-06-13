#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools


def read_image(filename):
    '''
    画像の読み込み
    '''
    img = cv2.imread(filename)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray_img


def triming_img(img):
    '''
    画像を中心から2/3のサイズで切り出し（トリミング）
    '''
    h, w = img.shape[:2]
    return img[int(h/3):int(h*2/3), int(w/3):int(w*2/3)]


def canny(img):
    '''
    画像内にある物体の輪郭を検出する
    '''
    img_preprocessed = cv2.cvtColor(cv2.GaussianBlur(img, (7, 7), 0), cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_preprocessed, threshold1=60, threshold2=80)
    return img_edges


def find_circle(img):
    '''
    円を検出する
    → 小さい光の円を検出できるようにしたい
    → 明るい状態でも検出できるようにしたい
    '''
    copy_img = np.copy(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.medianBlur(img, 9)
    img = cv2.blur(img, (9, 9))
    canny_img = cv2.Canny(img, threshold1=50, threshold2=100)

    # ret, th_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    # canny_img = canny(img)
    # hiyoko1
    # circles = cv2.HoughCircles(canny_img, cv2.HOUGH_GRADIENT, dp=2, minDist=250, minRadius=100, maxRadius=200)
    circles = cv2.HoughCircles(canny_img, cv2.HOUGH_GRADIENT, dp=2, minDist=10, minRadius=10, maxRadius=100)
    # hiyoko5
    # circles = cv2.HoughCircles(canny_img, cv2.HOUGH_GRADIENT, , 30, param1=0, param2=200, minRadius=0, maxRadius=0)
    # hiyokoneko4
    # circles = cv2.HoughCircles(th_img, cv2.HOUGH_GRADIENT, 3, 60, param1=30, param2=60, minRadius=0, maxRadius=80)
    # circles = cv2.HoughCircles(th_img, cv2.HOUGH_GRADIENT, dp=3, minDist=100, minRadius=50, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # 円の外枠
        cv2.circle(copy_img, (i[0], i[1]), i[2], (0, 0, 230), 2)
        # 円の中心
        cv2.circle(copy_img, (i[0], i[1]), 2, (0, 0, 230), 3)
    # cv2.imshow('preview', th_img)
    # cv2.imwrite('result.png', img)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()
    return copy_img


def find_lines(img, canny_img):
    '''
    画面内にある物体の直線を検知して、画面内にある交点を算出する
    '''
    # night
    # lines = cv2.HoughLines(canny_img, 2, np.pi/180, 230)
    # light
    lines = cv2.HoughLines(canny_img, 1, np.pi/180, 150)
    dots = []
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))

            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            dots.append([[x1, y1], [x2, y2]])

            img = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 240), 2)
    # cv2.imshow('preview', img)
    # cv2.waitKey(10000)
    # cv2.destroyAllWindows()
    # cv2.imwrite("./result/straight_night.png", img)
    return dots


def cal_intersection(canny_img, dots):
    '''
    直線の交点の座標を算出し、画像内に交点がある直線の組み合わを求める
    '''
    height, width = canny_img.shape
    intersection = []
    # 直線の組み合わせの数
    print(len(list(itertools.combinations(dots, 2))))
    for comb in itertools.combinations(dots, 2):
        x1, y1 = comb[0][0]
        x2, y2 = comb[0][1]
        x3, y3 = comb[1][0]
        x4, y4 = comb[1][1]
        # 交点が存在するかどうか判定
        if (x2-x1)*(y3-y2)+(y2-y1)*(x2-x3) * (x2-x1)*(y4-y2)+(y2-y1)*(x2-x4) < 0 \
        and (x4-x3)*(y2-y3)+(y4-y3)*(x3-x2) * (x4-x3)*(y1-y3)+(y4-y3)*(x3-x1) < 0:
            if (x2-x1) != 0 and (x4-x3) != 0:
                a1 = (y2-y1)/(x2-x1)
                a3 = (y4-y3)/(x4-x3)
                if (a1-a3) != 0:
                    x = (a1*x1-y1-a3*x3+y3)/(a1-a3)
                    y = (y2-y1)/(x2-x1)*(x-x1)+y1
                    # 画面内に交点があるかどうか判定
                    if 0 < y < height and 0 < x < width:
                        intersection.append((x, y))

    # 交点ができる直線の組み合わせ
    return intersection


def main():
    # img = cv2.imread('hiyoko1.jpeg')
    # img = cv2.imread('hiyoko1.jpg')
    img, gray_img = read_image('hiyoko6.jpeg')
    # img, gray_img = read_image('hiyokodog.jpg')
    # img, gray_img = read_image('hiyokodog1.jpg') # 難易度高
    # img, gray_img = read_image('hiyokoneko.jpeg')
    # img, gray_img = read_image('hiyokoneko4.jpg')
    # trim_img = triming_img(img)
    # img = cv2.blur(img, (9, 9))
    # canny_img = cv2.Canny(img, threshold1=50, threshold2=100)
    result = find_circle(img)
    # dots = find_lines(img, canny_img)
    # cv2.imwrite("./result.png", result)
    # print(len(cal_intersection(canny_img, dots)))
    cv2.imshow('preview', result)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
