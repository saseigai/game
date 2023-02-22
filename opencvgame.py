# coding:utf-8
import cv2
import  numpy as np
#from ultralytics import YOLO


#import pyautogui
#import torch
#tt=YOLO("yolov5n.pt")
def check_if_video_move():
    cap = cv2.VideoCapture('C:\\Users\\CB2\\Desktop\\pic\\test.mp4')
    ret,frame1=cap.read()
    gray1=cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    while True:
        # 读取当前帧
        ret, frame2 = cap.read()
        if not ret:
            break

        # 将当前帧转换为灰度图像
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # 计算两帧之间的差异
        diff = cv2.absdiff(gray1, gray2)

        # 对差异图像进行二值化处理
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]

        # 对二值化后的图像进行形态学处理，以去除噪声，并获取运动物体的边界框
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # 显示结果
        cv2.imshow('Motion Detection', frame2)
        if cv2.waitKey(1) == 1000:
            break

        # 更新上一帧的灰度图像
        gray1 = gray2

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


#check_if_video_move()




def gray_pic():
    # 读取图像
    img = cv2.imread("C:\\Users\\CB2\\Desktop\\deployment\\test2.jpg")
    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 创建一个空白图像，大小与输入图像相同
    contour_img = np.zeros(img.shape[:2], dtype=np.uint8)
    print(contour_img)
    cv2.drawContours(contour_img, contours, -1,(178,34,34),1)
    #cv2.drawContours(contour_img, contours,2000,(240,255,255),10)
    cv2.imshow('222.jpg',contour_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 保存灰度图像
    #cv2.imwrite("222.jpg", gray)
def check_line():
    img = cv2.imread('C:\\Users\\CB2\\Desktop\\deployment\\alley-5931413__480.jpg')

    # 将图像转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 进行Canny边缘检测
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 进行Hough变换
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # 找到指定长度和斜率的线段
    target_length =180
    target_slope = 0.1
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if abs(length - target_length) < 10:
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope - target_slope) < 0.1:
                print("Line found with length:", length, "and slope:", slope)
                break






def biaozhu():
    # 加载图像
    img = cv2.imread('C:\\Users\\CB2\\Desktop\\deployment\\test11.png')

    # 绘制矩形框
    cv2.rectangle(img, (100,100), (200, 200), (255,0, 255), 1)
#    print(cv2.rectangle(img, (100,200), (220,300), (255, 0, 0), 1))
    # 绘制文本标注
    cv2.putText(img, 'house', (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
 #   cv2.putText(img, 'Object', (200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # 显示图像
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def huidubiaozhu():
    # 加载图像
    img = cv2.imread('C:\\Users\\CB2\\Desktop\\deployment\\center.jpg')
    # 将图像转为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 进行边缘检测
    edges = cv2.Canny(gray,255,255)

    # 寻找轮廊
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个空白图像
  #  blank = np.zeros_like(img)
    blank = np.zeros((640, 320, 3),dtype=np.uint8 )
    # 在空白图像上绘制轮廊
    cv2.drawContours(blank, contours, -1, (173,255,47), 1)
    # 对轮廊进行标注
    #for i, contour in enumerate(contours):
    #    # 获取轮廊的外接矩形
    #    x, y, w, h = cv2.boundingRect(contour)
    #    print(x,y,w,h)
    #    # 在空白图像上绘制矩形框
    #    cv2.rectangle(blank, (x, y), (x + w, y + h), (0, 255, 0), 1)
#
    #    # 在矩形框上方添加标注文字
    #    cv2.putText(blank, f'Object {i + 1}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('result', blank)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def videos_to_picture():
    # 打开视频文件
    cap = cv2.VideoCapture('video.mp4')

    i = 0
    # 遍历视频每一帧
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 将帧转换成灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 保存图片
        cv2.imwrite(f'frame_{i}.jpg', gray)
        i += 1
    # 释放资源
    cap.release()

#gray_pic()
huidubiaozhu()
#biaozhu()
#check_line()

def  ttt():

    cap = cv2.VideoCapture('C:\\Users\\CB2\\Desktop\\pic\\test.mp4')  # 打开摄像头或视频文件
    prev_frame = None  # 前一帧图像
    threshold = 175  # 阈值
    kernel_size = 5
    iterations = 35  # 膨胀迭代次数
    min_area = 3000# 最小面积阈值

    while True:
        ret, frame = cap.read()  # 读取一帧图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像

        if prev_frame is None:
            prev_frame = gray.copy()  # 如果是第一帧，将其作为背景
        else:
            diff = cv2.absdiff(gray, prev_frame)  # 计算差异图像
            prev_frame = gray.copy()  # 更新前一帧

            thresh = cv2.threshold(diff, threshold, 150 , cv2.THRESH_BINARY)[1]  # 二值化处理
            dilated = np.zeros_like(thresh)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            dilated = cv2.dilate(thresh, kernel, iterations=iterations)  # 膨胀操作


            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    dilated_curcle=cv2.rectangle(dilated, (x, y), (x + w, y + h), (250,250,250), 10)  # 在原图上绘制矩形框
                    frame_curcle=cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255, 0), 3)


                    get_x_circle_center=(x+w/2,y+h/2)
                    pydirectinput.moveTo(100, 150)  # Move the mouse to the x, y coordinates 100, 150.
                    pydirectinput.click()  # Click the mouse at its current location.
                    pydirectinput.click(200, 220)  # Click the mouse at the x, y coordinates 200, 220.
                    pydirectinput.move(None,
                                       10)  # Move mouse 10 pixels down, that is, move the mouse relative to its current position.
                    pydirectinput.doubleClick()  # Double click the mouse at the
                    pydirectinput.press('esc')  # Simulate pressing the Escape key.
                    pydirectinput.keyDown('shift')
                    pydirectinput.keyUp('shift')

             #      print(dict(frame))
            cv2.imshow('frame', frame)  # 显示原图
            cv2.imshow('dilated', dilated)  # 显示膨胀后的前景区域

        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


#ttt()
