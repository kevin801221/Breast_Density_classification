import pandas as pd
import os
import cv2 as cv
from skimage.feature import canny
from skimage.feature import canny
from skimage.filters import sobel
from skimage.transform import probabilistic_hough_line
from skimage.draw import polygon
import numpy as np
import matplotlib.pylab as pylab
import copy
import pathlib
from tqdm import tqdm
from typing import List

def is_img_left(img):
    img_h, img_w= img.shape
    img_l = img[:,  :int(img_w/2)]
    img_r = img[:,  int(img_w/2):]
    if (img_r!=0).sum()>(img_l!=0).sum():
        return 0
    else: 
        return 1


def check_white_background(image):
    mean_value_of_corners = np.stack(
        [image[:500,:500], image[:500, -500:], image[-500:, -500:], image[-500:, :500]]
    ).mean()
    if mean_value_of_corners>127:
        return image*-1+255
    return image

def remove_text_label(image):
    # Remove the text label
    # Orientation has been done via RightOrient
    height, width = image.shape[:2]

    imgROI=image[:height//3, width*1//3:]
    _, binary = cv.threshold(imgROI, 63, 255, cv.THRESH_BINARY)
    # Look for the outline in the binarization plot
    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # OpenCV4~
    if not contours:
        return image
    ## drop countours close to the left and bottom boundary
    # contours_checked = contours
    contours_checked = []
    for ct in contours:        
        x,y,w,h = cv.boundingRect(ct)
        if x<5:
            continue
        if imgROI.shape[0]-y-h<5:
            continue
        contours_checked.append(ct)
    imgROI = cv.drawContours(imgROI, contours_checked, -1, color=0, thickness=-1)  # 第 i 个轮廓，内部填充
    image[:height//3, width*1//3:] = imgROI
    return image



def remove_white_borders(image, masks=None):
    # 3.2.1. Remove white border.
    # As shown in Figure 3, the image has a rough outer border with a white line, represented by a red line.
    # Because the white color in the mammogram also represents the tumor, it can lead to misdiagnosis of benign and malignant tumors.
    # Eliminate white lines to avoid misclassification by slightly using the crop function to crop the image area.
    # The image after removing the rough white line is shown in Figure 4.
    # Code for removing white borders from the image
    image = normalize_intensity(image).astype(np.uint8)
    _, binary = cv.threshold(image, 127, 255, cv.THRESH_OTSU + cv.THRESH_BINARY_INV)
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv.contourArea, reverse=True)  # All contours are sorted by area
    cnt = cnts[0]  # The largest contour in area
    rect = cv.boundingRect(cnt)  # The outline with the largest area is bounded by a vertical rectangle
    x, y, w, h = rect  # The coordinates of the upper left vertex of the rectangle (x,y), rectangle width w, height h
    # print("Vertical rectangle: (x,y)={}, (w,h)={}".format((x, y), (w, h)))
    if w*h/binary.shape[0]/binary.shape[1]<0.9:
        return image, masks, [ 0, 0]
    if masks is not None:
        for ii in range(len(masks)):
            assert image.shape==masks[ii].shape
    imgRWB = image[y+int(h*0.01):y+int(h*0.98), x+int(w*0.01):x+int(w*0.98)]
    if masks is not None:
        for ii in range(len(masks)):
            masks[ii] = masks[ii][y+int(h*0.01):y+int(h*0.98), x+int(w*0.01):x+int(w*0.98)]
    return imgRWB, masks, [y+int(h*0.01), x+int(w*0.01)]

def apply_canny(image, sigma=3):
    canny_img = canny(image, sigma)
    return sobel(canny_img)


def calculate_angle(line):
    diff_x = line[0][0]-line[1][0]
    diff_y = line[0][1]-line[1][1]
    return np.arctan2(diff_y, diff_x)*180/np.pi



def extract_lines(lines):
    def calculate_pos(l):
        return np.mean([x[0] for x in l]), np.mean([x[1] for x in l])
    def get_point(group_lines):
        tmp = []
        for gl in group_lines:
            tmp.extend(gl)
        return min([x[0] for x in tmp]), max([x[1] for x in tmp]), max([x[0] for x in tmp]), min([x[1] for x in tmp])
    def get_dist(l):
        return ((l[0]-l[2])**2+(l[1]-l[3])**2)**0.5
    
    line_pos = [calculate_pos(l) for l in lines]
    group = []
    group_id = []
    for l_id, l_pos in enumerate(line_pos):
        gotcha = 0
        for g, g_id in zip(group, group_id):
            if get_dist(
                ( 
                    get_point([lines[x] for x in g_id])[2], 
                    get_point([lines[x] for x in g_id])[3], 
                    lines[l_id][0][0], 
                    lines[l_id][0][1]
                ))>20 and \
            get_dist(
                ( 
                    get_point([lines[x] for x in g_id])[0], 
                    get_point([lines[x] for x in g_id])[1], 
                    lines[l_id][1][0], 
                    lines[l_id][1][1]
                ))>20:
                continue
            ang = calculate_angle((calculate_pos(g), l_pos))
            if ang>100 and ang<170:
                g.append(l_pos)
                g_id.append(l_id)
                gotcha = 1
                break
        if not gotcha:
            group.append([l_pos])
            group_id.append([l_id])

    group_lines = []
    for g_id in group_id:
        group_lines.append(
            [lines[l_id] for l_id in g_id]
        )
    group_lines.sort(key=lambda x: get_dist(get_point(x)))
    for gl in group_lines[::-1]:
        line = get_point(gl)
        dist = get_dist(get_point(gl))
        if line[0]/dist>1:
            continue
        if line[3]/dist>0.6:
            continue
        return (line[0], line[1]), (line[2], line[3])
    return []

def get_hough_lines(canny_img):
    lines_positioins = probabilistic_hough_line(
        canny_img,  
        threshold=20, 
        line_length=10,
        line_gap=3,
        seed=0
    )
    h, w = canny_img.shape
    # plt.imshow(canny_img, cmap=pylab.cm.gray)
    lines = []
    for line in lines_positioins:
        if calculate_angle(line)<93 or calculate_angle(line)>157:
            continue
        h_p = line[0][1]/h
        w_p = line[0][0]/w
        if calculate_angle(line)>80+(1-h_p)*90:
            continue
        # if h_p+w_p>.8:
        #     continue
        lines.append(line)
        # plt.plot(
        #     [line[0][0], line[1][0]],
        #     [line[0][1], line[1][1]],'r-'
        # )
    return lines
def remove_pectoral(lines, raw_image):
    img = copy.deepcopy(raw_image)
    for line in lines:
        rr, cc = polygon(
            [0, 0, line[1][1], line[0][1], line[0][1]],
            [0, line[1][0], line[1][0], line[0][0], 0]
        )
        raw_image[rr, cc] = 0
    return raw_image


def crop_w(image):
    tmp = (np.cumsum(image.sum(0))/image.sum()<0.99).sum()
    return image[:,:tmp+10]

def normalize_intensity(img):
    img = (img-img.min())/(img.max()-img.min())
    img = img*255
    return img.astype(int)


def process_image(img: np.array, sigma:float, masks: List[np.array]=None, skip_MR=False):
    ## 1. flip image
    image = copy.deepcopy(img)
    # image = check_white_background(image)
    flip = 0
    if not is_img_left(image):
        image = cv.flip(image, 1)
        if masks is not None:
            for ii in range(len(masks)):
                masks[ii] = cv.flip(masks[ii], 1)
        flip = 1

    ## 2. remove boundary and text
    image, masks, crop_shift = remove_white_borders(image, masks)
    image = remove_text_label(image) 
    raw_image = copy.deepcopy(image)

    if not skip_MR:
        ## 3. decrease resolution
        h, w = image.shape
        resize_portion = h/200
        image = cv.resize(image, (int(w/resize_portion), int(h/resize_portion)))
        resize_h, resize_w = image.shape
        image = crop_w(image)

        ## 4. enhace contrast
        # for ii in range(1):
        #     image[image<image.mean()] = image.mean()
        #     image =  (image-image.min())/(image.max()-image.min())
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image = clahe.apply(image)
        # plt.imshow(cl1, cmap=pylab.cm.gray)
        # plt.imshow(image, cmap=pylab.cm.gray)

        ## 5. get edge with canny
        canny_image = copy.deepcopy(image)
        # canny_image[canny_image>=0.5] = 1
        # canny_image[canny_image<0.5] = 0
        ##edge detection
        canny_image = apply_canny(canny_image, sigma)
        # raw_image = canny_image
        # plt.imshow(canny_image)

        ## 6. get the proper lines
        lines = get_hough_lines(canny_image)
        if lines:
            lines = extract_lines(lines)

        if len(lines):
            ## project line to raw resolution image
            line = (
                (
                    int(lines[0][0]/ resize_w*w),
                    int(lines[0][1]/ resize_h*h)
                ),
                (
                    int(lines[1][0]/ resize_w*w),
                    int(lines[1][1]/ resize_h*h)
                )
            )
            raw_image = remove_pectoral([line], raw_image)
    ## normalize pixel intensity
    raw_image = normalize_intensity(raw_image)

    ## flip back
    if flip:
        raw_image = cv.flip(raw_image, 1)
        if masks is not None:
            for ii in range(len(masks)):
                masks[ii] = cv.flip(masks[ii], 1)
    return raw_image, masks, crop_shift




if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-skip", "--skip", type=bool, default=False, help="skip the muscle removement process")
    args = vars(ap.parse_args())