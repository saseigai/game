def  partial_binary_video():

    cap = cv2.VideoCapture('test.mp4')

    # 获取视频的FPS和帧数
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建输出视频文件对象
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (640, 480))

    # 循环读取每一帧并进行局部二值化处理
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 中值滤波，测试只能是单数，目的去除噪点
        gray = cv2.medianBlur(gray, 7 )

        # 局部二值化处理，这里测试也是单数都行，偶数偶尔不行，35表示将图像分割成35*35个小单位，20代表常数
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 20)
        thresh = cv2.resize(thresh, (640, 480))  # 调整尺寸为输出视频的大小

        out.write(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))  # 将处理后的帧写入输出视频文件

        cv2.imshow('frame', thresh)  # 显示处理后的帧
        if cv2.waitKey(200) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
partial_binary_video()
