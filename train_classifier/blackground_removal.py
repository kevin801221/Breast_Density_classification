import cv2
import numpy as np
import os

def crop_image(image_path, output_path):
    # 讀取影像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 使用二值化找到非黑色區域
    _, binary = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)

    # 使用輪廓檢測找到邊緣
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果沒有找到任何輪廓，則跳過此影像
    if not contours:
        print(f"No contours found in image: {image_path}. Skipping...")
        return

    # 找到最大的輪廓（乳房的區域）
    max_contour = max(contours, key=cv2.contourArea)

    # 計算輪廓的邊界框
    x, y, w, h = cv2.boundingRect(max_contour)

    # 裁剪影像
    cropped_img = img[y:y+h, x:x+w]

    # 儲存裁剪後的影像
    cv2.imwrite(output_path, cropped_img)

# 指定輸入與輸出資料夾
input_folder = '/home/kevinluo/breast_density_classification/BD_data_newdis/test/level4'  # 將此路徑替換為你的輸入資料夾的路徑
output_folder = '/home/kevinluo/breast_density_classification/BD_data_newdis/test/level4'  # 將此路徑替換為你的輸出資料夾的路徑

# 確保輸出資料夾存在
os.makedirs(output_folder, exist_ok=True)

# 為資料夾內的每一個影像進行裁剪並儲存
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg'):  # 可以更換為你的影像檔案格式，例如：'.png', '.tif', 等等
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        crop_image(input_path, output_path)
