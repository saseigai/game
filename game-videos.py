def test1():
    # 读取图像和掩模图像
    img = cv2.imread('test2.jpg')
    mask = cv2.imread('test1.png', cv2.IMREAD_GRAYSCALE)

    # 确认输入图像和掩模图像的大小是否一致
    if img.shape != mask.shape:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # 确认输入图像和掩模图像的数据类型是否正确
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img)
    if mask.dtype != np.uint8:
        mask = cv2.convertScaleAbs(mask)

    # 确认调用二元操作函数时输入图像和掩模图像的数据类型是否一致
    if img.dtype != mask.dtype:
        mask = cv2.convertScaleAbs(mask)

    # 应用掩模
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # 显示结果
    cv2.imshow('Input Image', img)
    cv2.imshow('Mask', mask)
    cv2.imshow('Masked Image', masked_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
