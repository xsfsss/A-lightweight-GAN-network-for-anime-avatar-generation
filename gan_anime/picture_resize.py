#由于网络层次不能提取到全部特征，所以要将图片缩放
#Due to the inability of the network hierarchy to extract all features, the image needs to be scaled

import cv2
import os

input_folder = "path/to/input/folder"
output_folder = "path/to/output/folder"

# 遍历输入文件夹中的图片文件
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):

        # 打开图片
        img = cv2.imread(os.path.join(input_folder, filename))

        # 缩放图片
        img_resized = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

        # 保存图片
        output_filename = os.path.join(output_folder, filename)
        cv2.imwrite(output_filename, img_resized)
